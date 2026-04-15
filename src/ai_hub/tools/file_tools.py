from pathlib import Path
import logging

from ai_hub.config import CODING_WORKSPACE_ROOT
from ai_hub.logging_config import log_event, setup_logging


logger = logging.getLogger(__name__)
setup_logging()


class WorkspaceSecurityError(ValueError):
    def __init__(self, message: str, code: str = "workspace_access_denied") -> None:
        super().__init__(message)
        self.code = code
        self.user_message = message


SENSITIVE_PARTS = {
    ".env",
    ".git",
    ".ssh",
    "secrets",
    "secret",
    "token",
    "id_rsa",
    "id_ed25519",
}


def _workspace_root() -> Path:
    root = Path(CODING_WORKSPACE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def ensure_thread_workspace(thread_id: str) -> Path:
    workspace = _workspace_root() / thread_id
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace.resolve()


def _validate_relative_path(relative_path: str) -> Path:
    cleaned = (relative_path or "").strip().replace("\\", "/")
    if not cleaned:
        log_event(logger, "blocked_path", path=relative_path, reason="workspace_path_empty")
        raise WorkspaceSecurityError("Pfad darf nicht leer sein.", code="workspace_path_empty")
    if cleaned.startswith("/") or cleaned.startswith("~"):
        log_event(logger, "blocked_path", path=cleaned, reason="workspace_relative_paths_only")
        raise WorkspaceSecurityError(
            "Es sind nur relative Pfade im freigegebenen Workspace erlaubt.",
            code="workspace_relative_paths_only",
        )
    parts = [part for part in cleaned.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        log_event(logger, "blocked_path", path=cleaned, reason="workspace_path_escape_blocked")
        raise WorkspaceSecurityError(
            "Der angeforderte Pfad würde den erlaubten Workspace verlassen.",
            code="workspace_path_escape_blocked",
        )
    lowered = {part.lower() for part in parts}
    if lowered & SENSITIVE_PARTS:
        log_event(logger, "blocked_path", path=cleaned, reason="workspace_sensitive_path_blocked")
        raise WorkspaceSecurityError(
            "Zugriff auf sensible Dateien oder Verzeichnisse ist blockiert.",
            code="workspace_sensitive_path_blocked",
        )
    return Path(*parts)


def resolve_workspace_path(thread_id: str, relative_path: str) -> Path:
    workspace = ensure_thread_workspace(thread_id)
    target = (workspace / _validate_relative_path(relative_path)).resolve()
    if workspace not in target.parents and target != workspace:
        raise WorkspaceSecurityError(
            "Der Zielpfad liegt außerhalb des erlaubten Coding-Workspaces.",
            code="workspace_outside_root_blocked",
        )
    return target


def list_files(thread_id: str, relative_path: str = ".") -> list[dict]:
    target = ensure_thread_workspace(thread_id) if relative_path == "." else resolve_workspace_path(thread_id, relative_path)
    if not target.exists():
        log_event(logger, "list_files", thread_id=thread_id, path=relative_path, entries=0)
        return []

    entries: list[dict] = []
    for item in sorted(target.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower())):
        if item.name.lower() in SENSITIVE_PARTS:
            continue
        entries.append(
            {
                "name": item.name,
                "path": str(item.relative_to(ensure_thread_workspace(thread_id))),
                "is_dir": item.is_dir(),
            }
        )
    log_event(logger, "list_files", thread_id=thread_id, path=relative_path, entries=len(entries))
    return entries


def read_file(thread_id: str, relative_path: str) -> str:
    target = resolve_workspace_path(thread_id, relative_path)
    if not target.exists() or not target.is_file():
        log_event(logger, "read_file_missing", thread_id=thread_id, path=relative_path)
        raise FileNotFoundError(relative_path)
    content = target.read_text(encoding="utf-8")
    log_event(logger, "read_file", thread_id=thread_id, path=relative_path, bytes_read=len(content.encode("utf-8")))
    return content


def write_file(thread_id: str, relative_path: str, content: str) -> dict:
    target = resolve_workspace_path(thread_id, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    result = {
        "path": str(target.relative_to(ensure_thread_workspace(thread_id))),
        "bytes_written": len(content.encode("utf-8")),
    }
    log_event(logger, "write_file", thread_id=thread_id, path=result["path"], bytes_written=result["bytes_written"])
    return result


def make_directory(thread_id: str, relative_path: str) -> dict:
    target = resolve_workspace_path(thread_id, relative_path)
    target.mkdir(parents=True, exist_ok=True)
    result = {
        "path": str(target.relative_to(ensure_thread_workspace(thread_id))),
        "created": True,
    }
    log_event(logger, "make_directory", thread_id=thread_id, path=result["path"])
    return result


def delete_path(thread_id: str, relative_path: str) -> dict:
    target = resolve_workspace_path(thread_id, relative_path)
    if not target.exists():
        log_event(logger, "delete_path_missing", thread_id=thread_id, path=relative_path)
        raise FileNotFoundError(relative_path)

    if target.is_file():
        target.unlink()
        result = {
            "path": str(target.relative_to(ensure_thread_workspace(thread_id))),
            "deleted": True,
            "kind": "file",
        }
        log_event(logger, "delete_path", thread_id=thread_id, path=result["path"], kind="file")
        return result

    if any(target.iterdir()):
        log_event(logger, "blocked_path", thread_id=thread_id, path=relative_path, reason="workspace_non_empty_directory_delete_blocked")
        raise WorkspaceSecurityError(
            "Nicht-leere Ordner dürfen aktuell nicht gelöscht werden.",
            code="workspace_non_empty_directory_delete_blocked",
        )

    target.rmdir()
    result = {
        "path": str(target.relative_to(ensure_thread_workspace(thread_id))),
        "deleted": True,
        "kind": "directory",
    }
    log_event(logger, "delete_path", thread_id=thread_id, path=result["path"], kind="directory")
    return result
