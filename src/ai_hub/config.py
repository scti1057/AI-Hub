from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

BASE_DIR = Path(__file__).resolve().parents[2]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MANAGER_MODEL = os.getenv("MANAGER_MODEL", "gemma4:31b")
CODING_MODEL = os.getenv("CODING_MODEL", "qwen3-coder:30b")
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "qwen3:30b")
REVIEWER_MODEL = os.getenv(
    "REVIEWER_MODEL",
    "deepseek-r1:32b-qwen-distill-q4_K_M",
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = BASE_DIR / "logs"

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
