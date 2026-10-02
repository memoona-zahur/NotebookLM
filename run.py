import os

import uvicorn

if __name__ == "__main__":
    # Loopback by default: binding every interface on a dev machine exposes a
    # server with no auth to the whole network. Set HOST=0.0.0.0 deliberately
    # when you need to reach it from another machine.
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )