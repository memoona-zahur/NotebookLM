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
    db.migrate()
    db.ensure_default_session()
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

app.mount("/assets", StaticFiles(directory=config.ASSET_DIR), name="assets")