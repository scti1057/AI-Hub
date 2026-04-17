from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default

BASE_DIR = Path(__file__).resolve().parents[2]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MANAGER_MODEL = os.getenv("MANAGER_MODEL", "gemma4:31b")
CODING_MODEL = os.getenv("CODING_MODEL", "qwen3-coder:30b")
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "qwen3:30b")
REVIEWER_MODEL = os.getenv(
    "REVIEWER_MODEL",
    "qwen3-coder:30b",
)
REVIEWER_DEBUG_LOG_PROMPTS = _env_flag("REVIEWER_DEBUG_LOG_PROMPTS", True)
REVIEWER_DEBUG_LOG_MAX_CHARS = _env_int("REVIEWER_DEBUG_LOG_MAX_CHARS", 16000)
POST_CODING_REVIEW_DELAY_SECONDS = _env_float("POST_CODING_REVIEW_DELAY_SECONDS", 0.0)
MANAGER_AUTONOMOUS_MAX_STEPS = _env_int("MANAGER_AUTONOMOUS_MAX_STEPS", 20)
MANAGER_AUTONOMOUS_MAX_DEBUG_REPAIRS = _env_int("MANAGER_AUTONOMOUS_MAX_DEBUG_REPAIRS", 2)
CHAT_BACKGROUND_TIMEOUT_SECONDS = _env_float("CHAT_BACKGROUND_TIMEOUT_SECONDS", 330.0)
CHAT_THREAD_DUMP_ENABLED = _env_flag("CHAT_THREAD_DUMP_ENABLED", True)
CHAT_THREAD_DUMP_INTERVAL_SECONDS = _env_float("CHAT_THREAD_DUMP_INTERVAL_SECONDS", 15.0)
CHAT_THREAD_DUMP_MAX_CHARS = _env_int("CHAT_THREAD_DUMP_MAX_CHARS", 24000)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = BASE_DIR / "logs"
WEB_RELOAD = _env_flag("WEB_RELOAD", False)

APP_USERNAME = os.getenv("APP_USERNAME", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", "")

VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "")
VAPID_PRIVATE_KEY_PATH = os.getenv("VAPID_PRIVATE_KEY_PATH", "")
VAPID_SUBJECT = os.getenv("VAPID_SUBJECT", "")

DB_PATH = BASE_DIR / "data" / "memory" / "ai_hub.db"
CODING_WORKSPACE_ROOT = BASE_DIR / "data" / "workspaces" / "coding_agent"
APPROVAL_EXECUTION_TIMEOUT_SECONDS = int(
    os.getenv("APPROVAL_EXECUTION_TIMEOUT_SECONDS", "60")
)
MANAGER_LLM_PLANNING_ENABLED = _env_flag("MANAGER_LLM_PLANNING_ENABLED", True)
MANAGER_OLLAMA_ENABLED = _env_flag("MANAGER_OLLAMA_ENABLED", True)
PYTHON_SANDBOX_BACKEND = os.getenv("PYTHON_SANDBOX_BACKEND", "auto")

WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "auto")
WEB_SEARCH_ENABLED = _env_flag("WEB_SEARCH_ENABLED", True)
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_FETCH_PAGES = _env_flag("WEB_SEARCH_FETCH_PAGES", True)
WEB_SEARCH_FETCH_TOP_N = int(os.getenv("WEB_SEARCH_FETCH_TOP_N", "2"))
WEB_SEARCH_TIMEOUT_SECONDS = int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "15"))
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
