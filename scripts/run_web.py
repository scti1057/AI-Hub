import uvicorn

from ai_hub.config import WEB_RELOAD


if __name__ == "__main__":
    uvicorn.run(
        "ai_hub.web.app:app",
        host="127.0.0.1",
        port=8000,
        reload=WEB_RELOAD,
    )
