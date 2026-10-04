"""OCR support for scanned documents, and the installer for its language data.

PyMuPDF embeds Tesseract, so recognising text in a scan needs no external
`tesseract` binary - only the language files Tesseract reads. Those are not
bundled with PyMuPDF (they are tens of megabytes and there are dozens of
languages, so shipping a default is a packaging decision, not a library one).

    python -m app.ocr --install          # fetch eng.traineddata into data/tessdata
    python -m app.ocr --install deu fra  # add more languages
    python -m app.ocr --status           # what is available, and where

`--install` downloads from the official tesseract-ocr/tessdata_fast
repository. `tessdata_fast` rather than `tessdata_best`: it is ~4 MB per
language instead of ~12 MB, and for printed text on a clean scan the accuracy
difference is small enough that the speed and size win. `tessdata_best` is the
right trade for handwriting or a low-contrast photocopy - drop a file with the
same name into the folder and it is used instead.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

# tessdata_fast, the small models. See the module docstring for when to prefer
# tessdata_best instead.
BASE_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main"
# A 4 MB file should not take a minute; anything slower is a stalled connection
# and the timeout turns that into an error instead of a hang.
TIMEOUT_SECONDS = 120


def available_languages(tessdata_dir: Path) -> list[str]:
    """Language codes with their data file present, sorted."""
    if not tessdata_dir.is_dir():
        return []
    return sorted(path.stem for path in tessdata_dir.glob("*.traineddata"))


def missing_languages(tessdata_dir: Path) -> list[str]:
    from . import config

    wanted = [code for code in config.OCR_LANGUAGES.replace("+", " ").split() if code]
    have = set(available_languages(tessdata_dir))
    return [code for code in wanted if code not in have]


def install(languages: list[str], tessdata_dir: Path) -> int:
    """Download the language data. Returns a process exit code."""
    tessdata_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for code in languages:
        target = tessdata_dir / f"{code}.traineddata"
        if target.exists():
            print(f"  {code}: already present")
            continue
        url = f"{BASE_URL}/{code}.traineddata"
        print(f"  {code}: downloading from {url}")
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
                data = response.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            # One unreachable mirror should not look like a broken install, and
            # the language name in the message is what makes it actionable.
            print(f"  {code}: failed - {exc}")
            failures += 1
            continue
        if len(data) < 10_000:
            # A redirect to an HTML error page still arrives as a "success".
            print(f"  {code}: failed - response was {len(data)} bytes, not model data")
            failures += 1
            continue
        target.write_bytes(data)
        print(f"  {code}: saved {len(data) // 1024} KB to {target}")
    return 1 if failures else 0


def status(tessdata_dir: Path) -> int:
    from . import config

    have = available_languages(tessdata_dir)
    print(f"tessdata folder: {tessdata_dir}")
    if have:
        print(f"  languages available: {', '.join(have)}")
    else:
        print("  no language data found")
    wanted = [c for c in config.OCR_LANGUAGES.replace("+", " ").split() if c]
    print(f"configured (OCR_LANGUAGES): {', '.join(wanted) or 'none'}")
    absent = [code for code in wanted if code not in have]
    if absent:
        print(f"  missing: {', '.join(absent)}")
        print(f"  install with: python -m app.ocr --install {' '.join(absent)}")
        return 1
    print(f"  OCR is enabled (OCR_ENABLED={config.OCR_ENABLED})")
    return 0


def main(argv: list[str] | None = None) -> int:
    from . import config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--install",
        nargs="*",
        metavar="LANG",
        help="download language data (default: the configured OCR_LANGUAGES)",
    )
    parser.add_argument("--status", action="store_true", help="report what is available")
    parser.add_argument("--tessdata", type=Path, default=config.TESSDATA_DIR)
    args = parser.parse_args(argv)

    if args.install is not None:
        codes = args.install or [
            c for c in config.OCR_LANGUAGES.replace("+", " ").split() if c
        ]
        print(f"installing OCR language data into {args.tessdata}")
        return install(codes, args.tessdata)

    return status(args.tessdata)


if __name__ == "__main__":
    sys.exit(main())