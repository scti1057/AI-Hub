import contextvars
from datetime import datetime
import logging
import os
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ai_hub.config import BASE_DIR, LOG_LEVEL


_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("ai_hub_request_id", default="-")
_chat_log_dir: contextvars.ContextVar[str] = contextvars.ContextVar("ai_hub_chat_log_dir", default="")
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


def set_chat_log_dir(value: str) -> contextvars.Token[str]:
    return _chat_log_dir.set(value)


def reset_chat_log_dir(token: contextvars.Token[str]) -> None:
    _chat_log_dir.reset(token)


def get_chat_log_dir() -> str:
    return _chat_log_dir.get()


def create_chat_run_log_dir(thread_id: str, request_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = LOG_DIR / thread_id / f"{timestamp}_{request_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
    _append_chat_run_logs(logger.name, message)


def log_text_block(
    logger: logging.Logger,
    event: str,
    text: str,
    *,
    max_chars: int = 16000,
    **fields: object,
) -> None:
    payload = " ".join(f"{key}={_normalize_log_value(value)}" for key, value in fields.items())
    header = f"event={event}"
    if payload:
        header = f"{header} {payload}"
    body = text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"
    logger.info("%s\n%s", header, body)
    _append_chat_run_logs(logger.name, f"{header}\n{body}")


def log_thread_dump(
    logger: logging.Logger,
    event: str,
    *,
    max_chars: int = 24000,
    **fields: object,
) -> None:
    frames = sys._current_frames()
    thread_names = {thread.ident: thread.name for thread in threading.enumerate()}
    blocks: list[str] = []
    for ident, frame in frames.items():
        name = thread_names.get(ident, "unknown")
        stack = "".join(traceback.format_stack(frame))
        blocks.append(f"Thread id={ident} name={name}\n{stack}")
    dump = "\n\n".join(blocks) if blocks else "No active thread frames found."
    log_text_block(
        logger,
        event,
        dump,
        max_chars=max_chars,
        thread_count=len(frames),
        **fields,
    )


def _append_chat_run_logs(logger_name: str, message: str) -> None:
    request_id = get_request_id()
    chat_log_dir = get_chat_log_dir().strip()
    if request_id == "-" or not chat_log_dir:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    line = f"{timestamp} [{logger_name}] request_id={request_id} {message}\n"
    for filename in _target_run_log_files(logger_name):
        path = Path(chat_log_dir) / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _target_run_log_files(logger_name: str) -> list[str]:
    targets = ["app.log"]
    if logger_name.startswith(
        (
            "ai_hub.orchestration.workflow",
            "ai_hub.orchestration.delegation",
            "ai_hub.llm.manager_planner",
            "ai_hub.llm.ollama_client",
            "ai_hub.agents.explorer_agent",
            "ai_hub.agents.reviewer_agent",
            "ai_hub.agents.research_agent",
            "ai_hub.agents.coding_agent",
        )
    ):
        targets.append("exchange.log")
    if logger_name.startswith(("ai_hub.orchestration.workflow", "ai_hub.llm.manager_planner")):
        targets.append("manager.log")
    if logger_name.startswith("ai_hub.agents.explorer_agent"):
        targets.append("explorer.log")
    if logger_name.startswith("ai_hub.agents.reviewer_agent"):
        targets.append("reviewer.log")
    if logger_name.startswith("ai_hub.agents.research_agent"):
        targets.append("research.log")
    if logger_name.startswith("ai_hub.agents.coding_agent"):
        targets.append("coding.log")
    if logger_name.startswith("ai_hub.llm.ollama_client"):
        targets.append("ollama.log")
    if logger_name.startswith(
        (
            "ai_hub.tools.file_tools",
            "ai_hub.tools.code_runner",
            "ai_hub.tools.push_notify",
        )
    ):
        targets.append("tools.log")
    return targets
