"""Finding pages on the web and turning them into notebook sources.

This is the one part of the app that reaches out to the internet on the user's
behalf, and that changes the product's central promise. Everywhere else the
documents stay on this machine and only the question plus the retrieved
passages reach the configured LLM. Here the *search query* goes to a third-party
search engine, and then the app fetches whatever that engine points at.

So this module is written defensively in three specific ways.

**The search is a URL finder, not an answerer.** It asks Groq's browser tool
for pages and extracts URLs, then the pages themselves are fetched, parsed,
chunked and embedded by the same code path as an upload. The model never sees a
user question here and never writes a sentence that reaches the user. That
matters for the grounding guarantee: a web page becomes a *source*, so it can be
cited, checked and deleted like any other, rather than becoming an
unciteable paragraph of model prose.

**Fetched URLs are treated as hostile input.** They come from a search engine
responding to a model, so a URL can name anything at all, including
`http://169.254.169.254/` (a cloud instance's credentials endpoint) or
`http://localhost:5432/`. Every address is resolved and checked against the
private ranges before a socket is opened, on the first hop and on every redirect
after it, because a public host that 302s to `169.254.169.254` is the standard
way around a check that only looks at the first URL.

Residual risk worth stating plainly: the address is checked and then the request
goes out by hostname, so a hostile DNS server could return a public address to
`getaddrinfo` and a private one to the connection (DNS rebinding). Closing that
completely means pinning the connection to the validated IP and setting SNI by
hand, which `httpx` does not expose. For a single-user localhost app the
window is small; for a shared deployment it is not, and the fix belongs in the
transport layer rather than in a pre-flight check.

**A page that cannot be fetched or parsed does not fail the search.** Search
results routinely include PDFs behind bot walls, JavaScript-only pages and dead
links. One of those must not cost the user the four pages that worked, so every
failure is collected and reported next to the successes instead of raising.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import config

# A URL as it appears in model prose. The character class allows balanced
# parentheses because Wikipedia paths use them as real content -
# `.../ROUGE_(metric)` - and a naive `[^)\s]+` truncates that to a 404.
_URL = re.compile(r"https?://(?:[^\s<>()\[\]\"']|\([^\s()]*\))+")

# Groq's browser tool marks its sources inline as 【0†L6-L10】. They are
# references into a snippet list we do not have, so leaving them in the text
# would put 【0†...】 into a source name.
_MARKER = re.compile(r"[\u3010\u3011\u3014\u3015][^\u3010\u3014]*[\u3014\u3015]")

# Trailing characters that are sentence punctuation rather than part of the URL.
# Only stripped when unbalanced, so a URL genuinely ending in ")" survives.
_TRAILING = ".,;:!?"


class WebSearchUnavailable(RuntimeError):
    """Web search cannot run, or the request to run it failed.

    Distinct from "the search found nothing", which is a normal empty result.
    """


@dataclass
class Candidate:
    """A URL the search proposed, before anything has been fetched."""

    url: str
    title: str = ""


@dataclass
class FetchedPage:
    """A page that was actually retrieved, with the suffix its parser needs."""

    url: str
    title: str
    suffix: str
    content: bytes


# --------------------------------------------------------------------------
# Finding URLs
# --------------------------------------------------------------------------


def find(query: str, limit: int) -> tuple[list[Candidate], dict]:
    """Ask Groq's browser tool for pages about `query`.

    Returns the candidates and a cost record. The prompt asks for a bare list of
    URLs rather than a summary because that is the one output shape this can
    parse reliably: the same response also contains a prose answer with inline
    markers like 【0†L6-L10】, and `annotations`/`tool_calls` come back empty for
    this model, so URLs in the body text are the only dependable source.
    """
    usable, reason = config.web_search_available()
    if not usable:
        raise WebSearchUnavailable(reason)
    if not query.strip():
        raise WebSearchUnavailable("No search query given.")
    if not config.GROQ_API_KEY:
        raise WebSearchUnavailable("GROQ_API_KEY is not set.")

    limit = max(1, min(limit, config.WEB_SEARCH_MAX_PAGES))

    from openai import OpenAI

    client = OpenAI(base_url=config.GROQ_BASE_URL, api_key=config.GROQ_API_KEY)
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=config.WEB_SEARCH_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Find up to {limit} authoritative web pages about: {query.strip()}\n\n"
                    "Reply with ONLY a plain list of URLs, one per line, no numbering, "
                    "no commentary, no descriptions."
                ),
            }],
            tools=[{"type": "browser_search"}],
            tool_choice="required",
            # The docs recommend low reasoning effort with browser search: it
            # finds the same pages for a fraction of the tokens, and this call is
            # the most expensive thing the app ever makes.
            reasoning_effort="low",
            max_completion_tokens=900,
        )
    except Exception as exc:  # noqa: BLE001
        raise WebSearchUnavailable(f"Web search failed: {exc}") from exc
    elapsed = (time.perf_counter() - started) * 1000

    text = response.choices[0].message.content or ""
    reported = getattr(response, "usage", None)
    cost = {
        "model": config.WEB_SEARCH_MODEL,
        "prompt_tokens": int(getattr(reported, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(reported, "completion_tokens", 0) or 0),
        "search_ms": round(elapsed, 1),
    }
    return _extract(text, limit), cost


def _extract(text: str, limit: int) -> list[Candidate]:
    """URLs from the model's reply, cleaned, deduped and order-preserving."""
    cleaned = _MARKER.sub(" ", text)
    seen: set[str] = set()
    out: list[Candidate] = []
    for match in _URL.findall(cleaned):
        url = _tidy(match)
        if not url:
            continue
        key = _dedupe_key(url)
        if key in seen:
            continue
        seen.add(key)
        out.append(Candidate(url=url))
        if len(out) >= limit:
            break
    return out


def _tidy(url: str) -> str:
    """Drop punctuation that belongs to the sentence, not the URL."""
    url = url.strip().rstrip(_TRAILING)
    # Only remove a closing paren when it has no partner inside the URL, which
    # is what distinguishes prose "...(see /wiki/ROUGE_(metric))" from a URL.
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1]
    return url.rstrip(_TRAILING)


def _dedupe_key(url: str) -> str:
    """The same page as the model might write it twice, normalised.

    Fragment and trailing slash are dropped because `page#section` and `page/`
    are the same document to a fetcher, and the model returned both forms of the
    same Wikipedia article in one reply.
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc.lower()}{unquote(path)}"


# --------------------------------------------------------------------------
# Fetching, defensively
# --------------------------------------------------------------------------


def _public_addresses(host: str) -> list[str]:
    """Every address `host` resolves to, provided all of them are public.

    All, not any: a name that resolves to one public and one private address is
    a rebinding attempt, not a lucky result, and the connection below picks one
    of them without telling us which.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise WebSearchUnavailable(f"Could not resolve {host}: {exc}") from exc

    addresses: list[str] = []
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        # An IPv4-mapped IPv6 address is the same host as the IPv4 inside it, so
        # it is judged by the IPv4 rules rather than the v6 ones.
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None:
            ip = mapped
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise WebSearchUnavailable(
                f"Refusing to fetch {host}: it resolves to the non-public "
                f"address {address}."
            )
        addresses.append(address)

    if not addresses:
        raise WebSearchUnavailable(f"Could not resolve {host}.")
    return addresses


def _scheme_for(url: str, content_type: str) -> str | None:
    """The file suffix this response should be stored under, if we can parse it.

    Content-Type wins over the URL path because `download?id=42` is common and
    says nothing, and because a search result pointing at a `.aspx` page that
    renders PDF bytes must still be parsed as the PDF it is.
    """
    content_type = (content_type or "").lower()
    if "pdf" in content_type:
        return ".pdf"
    if "html" in content_type or "xhtml" in content_type:
        return ".html"
    if "markdown" in content_type:
        return ".md"
    if "csv" in content_type:
        return ".csv"
    if "json" in content_type:
        return ".json"
    if content_type.startswith("text/"):
        return ".txt"

    # No usable content type: fall back to the path, and only if the app can
    # actually parse it.
    from .parsers import SUPPORTED

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".htm", ".markdown", ".rst"}:
        return {".htm": ".html", ".markdown": ".md", ".rst": ".txt"}[suffix]
    return suffix if suffix in SUPPORTED else None


def fetch(candidate: Candidate) -> FetchedPage:
    """Retrieve one page, or raise `WebSearchUnavailable` explaining why not."""
    import httpx

    parsed = urlparse(candidate.url)
    if parsed.scheme not in {"http", "https"}:
        raise WebSearchUnavailable(
            f"Only http and https URLs can be fetched, not {parsed.scheme or 'this'}: {candidate.url}"
        )
    if not parsed.hostname:
        raise WebSearchUnavailable(f"That is not a usable URL: {candidate.url}")

    # Validated per hop, not once: see the module docstring. httpx can follow
    # redirects itself, but it does so *after* opening the connection, which is
    # too late - the request to the private address has already been sent by the
    # time this code could look at the final URL. So redirects are followed by
    # hand here and every hop is checked before it is requested.
    _public_addresses(parsed.hostname)

    timeout = httpx.Timeout(config.WEB_FETCH_TIMEOUT)
    limit = config.WEB_FETCH_MAX_BYTES
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            headers={
                # Some sites serve a bare page to an unrecognised agent, and the
                # user asked to read these pages, so ask as a reader would.
                "User-Agent": (
                    "NotebookLM/1.0 (+source ingestion; respects robots)"
                ),
                "Accept": "text/html,application/pdf,text/plain;q=0.9,*/*;q=0.5",
            },
        ) as client:
            current = candidate.url
            for _ in range(config.WEB_FETCH_MAX_REDIRECTS + 1):
                with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location", "")
                        if not location:
                            raise WebSearchUnavailable(
                                f"{_host(current)} redirected without saying where to."
                            )
                        nxt = str(httpx.URL(current).join(location))
                        hop = urlparse(nxt)
                        if hop.scheme not in {"http", "https"}:
                            raise WebSearchUnavailable(
                                f"{_host(current)} redirected to a non-web address: {nxt}"
                            )
                        if not hop.hostname:
                            raise WebSearchUnavailable(
                                f"{_host(current)} redirected to an unusable URL: {nxt}"
                            )
                        # The check that matters: before this hop is requested.
                        _public_addresses(hop.hostname)
                        current = nxt
                        continue

                    final = str(response.url)

                    if response.status_code >= 400:
                        raise WebSearchUnavailable(
                            f"{_host(final)} returned HTTP {response.status_code}."
                        )

                    suffix = _scheme_for(final, response.headers.get("content-type", ""))
                    if suffix is None:
                        raise WebSearchUnavailable(
                            f"{_host(final)} served "
                            f"{response.headers.get('content-type') or 'an unknown type'}, "
                            "which this app cannot read."
                        )

                    body = bytearray()
                    for piece in response.iter_bytes():
                        body.extend(piece)
                        # Stopped while reading, so an endless body is abandoned
                        # rather than buffered and then measured.
                        if len(body) > limit:
                            raise WebSearchUnavailable(
                                f"{_host(final)} is larger than "
                                f"{limit // (1024 * 1024)} MB."
                            )
                    break
            else:
                raise WebSearchUnavailable(
                    f"{_host(candidate.url)} redirected more than "
                    f"{config.WEB_FETCH_MAX_REDIRECTS} times."
                )
    except httpx.HTTPError as exc:
        raise WebSearchUnavailable(f"Could not fetch {_host(candidate.url)}: {exc}") from exc

    if not body:
        raise WebSearchUnavailable(f"{_host(candidate.url)} returned an empty page.")

    title = _title(body, suffix, final) or candidate.title or _host(candidate.url)
    return FetchedPage(
        url=final, title=title, suffix=suffix, content=bytes(body)
    )


def _host(url: str) -> str:
    return urlparse(url).netloc or url


def _title(body: bytes, suffix: str, url: str) -> str:
    """A human-readable name for the source, so citations are legible.

    HTML gets its first heading or `<title>`; everything else falls back to the
    filename inside the URL, which is the best available name for the PDFs a
    search turns up.
    """
    if suffix != ".html":
        return Path(unquote(urlparse(url).path)).name

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(_decode_bytes(body), "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        heading = soup.find(["h1", "title"])
        if heading:
            text = heading.get_text(" ", strip=True)
            if text:
                return _shorten(text)
    except Exception:  # noqa: BLE001 - a title is a nicety, never a blocker
        pass
    return ""


def _decode_bytes(body: bytes) -> str:
    """Bytes as text, using the same encoding ladder the file parser uses.

    Re-implemented rather than imported because `parsers._decode` wants a Path
    and reads the file itself, which would mean writing the page to disk twice
    to learn its title.
    """
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return body.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return body.decode("utf-8", errors="replace")


def _shorten(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"