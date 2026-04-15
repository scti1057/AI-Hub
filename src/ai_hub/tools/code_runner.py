import os
import shutil
import subprocess
import sys
import logging
from pathlib import Path

from ai_hub.config import APPROVAL_EXECUTION_TIMEOUT_SECONDS, PYTHON_SANDBOX_BACKEND
from ai_hub.logging_config import log_event, setup_logging
from ai_hub.tools.file_tools import ensure_thread_workspace, resolve_workspace_path

logger = logging.getLogger(__name__)
setup_logging()

class ExecutionPolicyError(ValueError):
    def __init__(self, message: str, code: str = "execution_policy_denied") -> None:
        super().__init__(message)
        self.code = code
        self.user_message = message


ALLOWED_MODULES = {"pytest", "unittest"}


def get_sandbox_backend() -> str:
    if PYTHON_SANDBOX_BACKEND == "bwrap":
        return "bwrap"
    if PYTHON_SANDBOX_BACKEND == "subprocess":
        return "subprocess"
    if shutil.which("bwrap"):
        return "bwrap"
    return "subprocess"


def build_python_execution_request(
    thread_id: str,
    argv: list[str],
    rationale: str,
) -> dict:
    if not argv:
        raise ExecutionPolicyError(
            "Leerer Python-Befehl ist nicht erlaubt.",
            code="execution_empty_command",
        )

    preview = "python " + " ".join(argv)
    command = {
        "thread_id": thread_id,
        "argv": argv,
        "preview": preview,
        "rationale": rationale,
    }
    log_event(logger, "execution_request_built", thread_id=thread_id, command_preview=preview)
    return command


def _validate_python_argv(thread_id: str, argv: list[str]) -> list[str]:
    if not argv:
        raise ExecutionPolicyError(
            "Keine Python-Argumente übergeben.",
            code="execution_missing_arguments",
        )
    if argv[0] == "-c":
        raise ExecutionPolicyError(
            "Inline-Python ist nicht erlaubt.",
            code="execution_inline_python_blocked",
        )
    if argv[0] == "-m":
        if len(argv) < 2 or argv[1] not in ALLOWED_MODULES:
            raise ExecutionPolicyError(
                "Nur ausdrücklich freigegebene Python-Module dürfen gestartet werden.",
                code="execution_module_not_allowed",
            )
        return argv

    script_path = resolve_workspace_path(thread_id, argv[0])
    if script_path.suffix != ".py":
        raise ExecutionPolicyError(
            "Es dürfen nur Python-Dateien aus dem Sandbox-Workspace ausgeführt werden.",
            code="execution_only_python_files_allowed",
        )

    safe_argv = [str(Path(argv[0]))]
    safe_argv.extend(argv[1:])
    return safe_argv


def execute_python_approval(command: dict) -> dict:
    thread_id = command["thread_id"]
    argv = _validate_python_argv(thread_id, list(command["argv"]))
    workspace = ensure_thread_workspace(thread_id)
    backend = get_sandbox_backend()
    log_event(logger, "execution_started", thread_id=thread_id, command_preview=command.get("preview"), backend=backend)

    if backend == "bwrap":
        try:
            completed, commandline = _run_with_bwrap(workspace, argv)
            if _should_fallback_from_bwrap(completed):
                fallback = _run_with_subprocess(workspace, argv)
                result = {
                    "command": [sys.executable, *argv],
                    "returncode": fallback.returncode,
                    "stdout": fallback.stdout,
                    "stderr": f"bubblewrap fallback: {completed.stderr}\n{fallback.stderr}",
                    "workspace": str(workspace),
                    "sandbox_backend": "subprocess_fallback",
                }
                log_event(
                    logger,
                    "execution_finished",
                    thread_id=thread_id,
                    backend=result["sandbox_backend"],
                    returncode=result["returncode"],
                    stdout=result["stdout"],
                    stderr=result["stderr"],
                )
                return result
            result = {
                "command": commandline,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "workspace": str(workspace),
                "sandbox_backend": "bwrap",
            }
            log_event(
                logger,
                "execution_finished",
                thread_id=thread_id,
                backend=result["sandbox_backend"],
                returncode=result["returncode"],
                stdout=result["stdout"],
                stderr=result["stderr"],
            )
            return result
        except (OSError, subprocess.SubprocessError) as exc:
            completed = _run_with_subprocess(workspace, argv)
            result = {
                "command": [sys.executable, *argv],
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": f"bubblewrap fallback: {exc}\n{completed.stderr}",
                "workspace": str(workspace),
                "sandbox_backend": "subprocess_fallback",
            }
            log_event(
                logger,
                "execution_finished",
                thread_id=thread_id,
                backend=result["sandbox_backend"],
                returncode=result["returncode"],
                stdout=result["stdout"],
                stderr=result["stderr"],
            )
            return result

    completed = _run_with_subprocess(workspace, argv)
    result = {
        "command": [sys.executable, *argv],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "workspace": str(workspace),
        "sandbox_backend": "subprocess",
    }
    log_event(
        logger,
        "execution_finished",
        thread_id=thread_id,
        backend=result["sandbox_backend"],
        returncode=result["returncode"],
        stdout=result["stdout"],
        stderr=result["stderr"],
    )
    return result


def validate_python_execution_request(command: dict) -> dict:
    thread_id = command["thread_id"]
    argv = _validate_python_argv(thread_id, list(command["argv"]))
    workspace = ensure_thread_workspace(thread_id)

    details = {
        "thread_id": thread_id,
        "argv": argv,
        "workspace": str(workspace),
    }

    if argv[0] == "-m":
        for candidate in argv[2:]:
            candidate_path = resolve_workspace_path(thread_id, candidate)
            if not candidate_path.exists():
                log_event(logger, "execution_request_blocked", thread_id=thread_id, command_preview=command.get("preview"), reason="execution_target_missing", path=candidate)
                raise ExecutionPolicyError(
                    f"Die angeforderte Testdatei `{candidate}` existiert im Workspace noch nicht.",
                    code="execution_target_missing",
                )
        log_event(logger, "execution_request_validated", thread_id=thread_id, command_preview=command.get("preview"))
        return details

    script_path = resolve_workspace_path(thread_id, argv[0])
    if not script_path.exists():
        log_event(logger, "execution_request_blocked", thread_id=thread_id, command_preview=command.get("preview"), reason="execution_target_missing", path=argv[0])
        raise ExecutionPolicyError(
            f"Die angeforderte Python-Datei `{argv[0]}` existiert im Workspace noch nicht.",
            code="execution_target_missing",
        )
    log_event(logger, "execution_request_validated", thread_id=thread_id, command_preview=command.get("preview"))
    return details


def _run_with_subprocess(workspace: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *argv],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=APPROVAL_EXECUTION_TIMEOUT_SECONDS,
        check=False,
    )


def _run_with_bwrap(
    workspace: Path,
    argv: list[str],
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    python_executable = Path(sys.executable).resolve()
    python_root = python_executable.parents[1]
    commandline = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--hostname",
        "ai-hub-runner",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/tmp",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        str(python_root),
        str(python_root),
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(workspace),
        "/workspace",
        "--chdir",
        "/workspace",
        str(python_executable),
        *argv,
    ]
    completed = subprocess.run(
        commandline,
        capture_output=True,
        text=True,
        timeout=APPROVAL_EXECUTION_TIMEOUT_SECONDS,
        check=False,
        env={},
    )
    return completed, commandline


def _should_fallback_from_bwrap(completed: subprocess.CompletedProcess[str]) -> bool:
    stderr = (completed.stderr or "").lower()
    return completed.returncode != 0 and (
        "operation not permitted" in stderr
        or "creating new namespace" in stderr
        or "no permissions to create new namespace" in stderr
        or "permission denied" in stderr
        or "failed to create netlink_route socket" in stderr
        or "failed to create netlink route socket" in stderr
    )
