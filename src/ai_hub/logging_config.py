import contextvars
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ai_hub.config import BASE_DIR, LOG_LEVEL


_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("ai_hub_request_id", default="-")
_configured = False

LOG_DIR = BASE_DIR / "logs"
APP_LOG = LOG_DIR / "app.log"
MANAGER_LOG = LOG_DIR / "manager.log"
EXECUTION_LOG = LOG_DIR / "execution.log"
MAX_BYTES = int(os.getenv("LOG_FILE_MAX_BYTES", str(2 * 1024 * 1024)))
BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class PrefixFilter(logging.Filter):
    def __init__(self, prefixes: tuple[str, ...]) -> None:
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self.prefixes)


def set_request_id(value: str) -> contextvars.Token[str]:
    return _request_id.set(value)


def reset_request_id(token: contextvars.Token[str]) -> None:
    _request_id.reset(token)


def get_request_id() -> str:
    return _request_id.get()


def setup_logging() -> None:
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] request_id=%(request_id)s %(message)s"
    )
    request_filter = RequestIdFilter()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    console.addFilter(request_filter)

    app_file = RotatingFileHandler(APP_LOG, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
    app_file.setLevel(level)
    app_file.setFormatter(formatter)
    app_file.addFilter(request_filter)

    manager_file = RotatingFileHandler(MANAGER_LOG, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
    manager_file.setLevel(level)
    manager_file.setFormatter(formatter)
    manager_file.addFilter(request_filter)
    manager_file.addFilter(PrefixFilter(("ai_hub.orchestration.workflow", "ai_hub.llm.manager_planner")))

    execution_file = RotatingFileHandler(
        EXECUTION_LOG,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    execution_file.setLevel(level)
    execution_file.setFormatter(formatter)
    execution_file.addFilter(request_filter)
    execution_file.addFilter(
        PrefixFilter(
            (
                "ai_hub.tools.code_runner",
                "ai_hub.tools.file_tools",
                "ai_hub.agents.coding_agent",
                "ai_hub.tools.push_notify",
            )
        )
    )

    root_logger = logging.getLogger("ai_hub")
    root_logger.setLevel(level)
    root_logger.propagate = False
    root_logger.handlers.clear()
    root_logger.addHandler(console)
    root_logger.addHandler(app_file)
    root_logger.addHandler(manager_file)
    root_logger.addHandler(execution_file)

    _configured = True


def _normalize_log_value(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).replace("\n", "\\n")
    if len(text) > 240:
        text = text[:237] + "..."
    return text


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    payload = " ".join(f"{key}={_normalize_log_value(value)}" for key, value in fields.items())
    message = f"event={event}"
    if payload:
        message = f"{message} {payload}"
    logger.info(message)
