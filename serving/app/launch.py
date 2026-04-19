from __future__ import annotations

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.service_host,
        port=settings.service_port,
        workers=max(1, settings.web_concurrency),
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
