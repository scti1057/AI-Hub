import os
import shutil
import subprocess
import sys
import logging
from pathlib import Path

from ai_hub.config import APPROVAL_EXECUTION_TIMEOUT_SECONDS, PYTHON_SANDBOX_BACKEND
from ai_hub.logging_config import log_event, setup_logging
from ai_hub.tools.file_tools import ensure_thread_workspace, resolve_workspace_path
from ai_hub.tools.dependency_tools import (
    WORKSPACE_REQUIREMENTS_FILE,
    WORKSPACE_VENV_DIR,
    workspace_requirements_path,
    workspace_venv_path,
)

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


def build_python_package_install_request(
    thread_id: str,
    packages: list[str],
    rationale: str,
) -> dict:
    normalized_packages = [package.strip() for package in packages if package and package.strip()]
    if not normalized_packages:
        raise ExecutionPolicyError(
            "Leerer Paket-Installationsbefehl ist nicht erlaubt.",
            code="execution_empty_package_install",
        )
    preview = "python -m pip install " + " ".join(normalized_packages)
    command = {
        "kind": "pip_install",
        "thread_id": thread_id,
        "packages": normalized_packages,
        "preview": preview,
        "rationale": rationale,
        "requirements_file": WORKSPACE_REQUIREMENTS_FILE,
        "venv_dir": WORKSPACE_VENV_DIR,
    }
    log_event(logger, "package_install_request_built", thread_id=thread_id, command_preview=preview)
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
    if command.get("kind") == "pip_install":
        return execute_package_install_approval(command)
    thread_id = command["thread_id"]
    argv = _validate_python_argv(thread_id, list(command["argv"]))
    workspace = ensure_thread_workspace(thread_id)
    backend = get_sandbox_backend()
    python_executable = _workspace_python_executable(thread_id)
    log_event(logger, "execution_started", thread_id=thread_id, command_preview=command.get("preview"), backend=backend)

    if backend == "bwrap":
        try:
            completed, commandline = _run_with_bwrap(workspace, argv, python_executable=python_executable)
            if _should_fallback_from_bwrap(completed):
                fallback = _run_with_subprocess(workspace, argv, python_executable=python_executable)
                result = {
                    "command": [str(python_executable), *argv],
                    "command_kind": "python_execution",
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
                "command_kind": "python_execution",
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
            completed = _run_with_subprocess(workspace, argv, python_executable=python_executable)
            result = {
                "command": [str(python_executable), *argv],
                "command_kind": "python_execution",
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

    completed = _run_with_subprocess(workspace, argv, python_executable=python_executable)
    result = {
        "command": [str(python_executable), *argv],
        "command_kind": "python_execution",
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
    if command.get("kind") == "pip_install":
        return validate_python_package_install_request(command)
    thread_id = command["thread_id"]
    argv = _validate_python_argv(thread_id, list(command["argv"]))
    workspace = ensure_thread_workspace(thread_id)

    details = {
        "thread_id": thread_id,
        "argv": argv,
        "workspace": str(workspace),
    }

    if argv[0] == "-m":
        for candidate in _workspace_like_module_args(argv[2:]):
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


def _workspace_like_module_args(args: list[str]) -> list[str]:
    candidates: list[str] = []
    for candidate in args:
        text = str(candidate).strip()
        if not text or text.startswith("-"):
            continue
        if text.endswith(".py") or "/" in text or "\\" in text:
            candidates.append(text)
    return candidates


def validate_python_package_install_request(command: dict) -> dict:
    thread_id = command["thread_id"]
    packages = list(command.get("packages") or [])
    if not packages:
        raise ExecutionPolicyError(
            "Es wurden keine Pakete für die Installation angegeben.",
            code="execution_missing_package_arguments",
        )
    for package in packages:
        cleaned = str(package).strip()
        if not cleaned:
            raise ExecutionPolicyError(
                "Leere Paketnamen sind nicht erlaubt.",
                code="execution_invalid_package_name",
            )
        if any(token in cleaned for token in (";", "&", "|", "$", "`", ">", "<")):
            raise ExecutionPolicyError(
                "Unsichere Paketangaben sind nicht erlaubt.",
                code="execution_invalid_package_name",
            )
    workspace = ensure_thread_workspace(thread_id)
    return {
        "thread_id": thread_id,
        "packages": packages,
        "workspace": str(workspace),
        "requirements_file": str(workspace_requirements_path(thread_id).relative_to(workspace)),
        "venv_dir": str(workspace_venv_path(thread_id).relative_to(workspace)),
    }


def execute_package_install_approval(command: dict) -> dict:
    details = validate_python_package_install_request(command)
    thread_id = details["thread_id"]
    workspace = ensure_thread_workspace(thread_id)
    packages = details["packages"]
    venv_path = workspace_venv_path(thread_id)
    python_executable = _ensure_workspace_venv(venv_path)
    pip_install = subprocess.run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            *packages,
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=APPROVAL_EXECUTION_TIMEOUT_SECONDS,
        check=False,
    )
    requirements_file = workspace_requirements_path(thread_id)
    if pip_install.returncode == 0:
        existing = []
        if requirements_file.exists():
            existing = [
                line.strip()
                for line in requirements_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        merged = existing[:]
        existing_normalized = {line.lower().replace("_", "-") for line in merged}
        for package in packages:
            normalized = package.lower().replace("_", "-")
            if normalized not in existing_normalized:
                merged.append(package)
                existing_normalized.add(normalized)
        requirements_file.write_text("".join(f"{line}\n" for line in merged), encoding="utf-8")
    return {
        "command": [str(python_executable), "-m", "pip", "install", *packages],
        "command_kind": "pip_install",
        "returncode": pip_install.returncode,
        "stdout": pip_install.stdout,
        "stderr": pip_install.stderr,
        "workspace": str(workspace),
        "sandbox_backend": "subprocess",
        "requirements_file": str(requirements_file.relative_to(workspace)),
        "venv_dir": str(venv_path.relative_to(workspace)),
        "packages": packages,
    }


def _run_with_subprocess(
    workspace: Path,
    argv: list[str],
    *,
    python_executable: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(python_executable), *argv],
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
    return _run_with_bwrap(workspace, argv, python_executable=Path(sys.executable).resolve())


def _run_with_bwrap(
    workspace: Path,
    argv: list[str],
    *,
    python_executable: Path,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
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


def _workspace_python_executable(thread_id: str) -> Path:
    venv_path = workspace_venv_path(thread_id)
    candidate = venv_path / "bin" / "python"
    if candidate.exists():
        return candidate.resolve()
    return Path(sys.executable).resolve()


def _python_has_pip(python_executable: Path, workspace: Path) -> bool:
    probe = subprocess.run(
        [str(python_executable), "-m", "pip", "--version"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=APPROVAL_EXECUTION_TIMEOUT_SECONDS,
        check=False,
    )
    return probe.returncode == 0


def _bootstrap_workspace_pip(python_executable: Path, workspace: Path) -> None:
    ensurepip = subprocess.run(
        [str(python_executable), "-m", "ensurepip", "--upgrade"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=APPROVAL_EXECUTION_TIMEOUT_SECONDS,
        check=False,
    )
    if ensurepip.returncode != 0 or not _python_has_pip(python_executable, workspace):
        error_message = ensurepip.stderr.strip() or ensurepip.stdout.strip()
        raise ExecutionPolicyError(
            error_message or "Im Workspace-Virtualenv konnte pip nicht bereitgestellt werden.",
            code="execution_workspace_pip_unavailable",
        )


def _ensure_workspace_venv(venv_path: Path) -> Path:
    python_executable = venv_path / "bin" / "python"
    workspace = venv_path.parent
    if python_executable.exists():
        if not _python_has_pip(python_executable, workspace):
            _bootstrap_workspace_pip(python_executable, workspace)
        return python_executable.resolve()
    created = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_path)],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=APPROVAL_EXECUTION_TIMEOUT_SECONDS,
        check=False,
    )
    if created.returncode != 0 or not python_executable.exists():
        raise ExecutionPolicyError(
            created.stderr.strip() or "Das Workspace-Virtualenv konnte nicht erstellt werden.",
            code="execution_venv_creation_failed",
        )
    if not _python_has_pip(python_executable, workspace):
        _bootstrap_workspace_pip(python_executable, workspace)
    return python_executable.resolve()


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
