"""Application assembly.

This file is meant to be short. Its whole job is to construct the app, run the
startup and shutdown work, mount the built frontend, and attach the routers.

That it fits on one screen is the point. When the entry point lists what the app
*is* - four routers, one lifespan, one static mount - a reader can hold the
shape of the system in their head before reading any of it. The earlier version
of this file was 450 lines and mixed all of those concerns with the retrieval
policy, which made the shape impossible to see without reading everything.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import config, db
from .routes import chat, meta, sessions, sources


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Brings a fresh volume up to head and leaves an existing one where it
    # already is, so neither needs a manual migration step before first use.
    try:
        db.migrate()
        db.ensure_default_session()
    except Exception as exc:  # noqa: BLE001 - re-raised below with the one thing to do
        raise RuntimeError(db.unreachable_message(exc)) from exc
    yield
    # Without this the pool's connections are only reclaimed when the process
    # dies, which matters because uvicorn runs several workers under a reload.
    db.close_pool()


app = FastAPI(title="Local NotebookLM", version="2.0.0", lifespan=lifespan)

for module in (sessions, sources, chat, meta):
    app.include_router(module.router)

# The built bundle references its assets by absolute path (/assets/app.js), so
# they are served from /assets as well as from /static. Missing on purpose would
# mean a blank page with no explanation, so this fails loudly at boot instead.
if not config.ASSET_DIR.is_dir():
    raise RuntimeError(
        f"No built frontend at {config.ASSET_DIR}. "
        "Run `npm install && npm run build` in frontend/."
    )



class RevalidatingStatic(StaticFiles):
    """Static files that the browser must re-check before reusing.

    The bundle keeps one filename (`assets/app.js`) on every build. With no
    Cache-Control a browser may reuse its copy for a heuristic period after an
    upgrade, and the page then runs the old interface against the new API. With
    `no-cache` it still keeps the file but asks first; the ETag answers with a
    304 and nothing is downloaded unless the file changed.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/assets", RevalidatingStatic(directory=config.ASSET_DIR), name="assets")