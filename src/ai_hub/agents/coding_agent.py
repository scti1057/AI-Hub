import json
import logging
import re
import shlex
from pathlib import Path

from ai_hub.config import CODING_DEBUG_LOG_MAX_CHARS, CODING_DEBUG_LOG_PROMPTS
from ai_hub.language_policy import LanguagePolicy
from ai_hub.logging_config import log_event, log_text_block, setup_logging
from ai_hub.llm.model_router import ModelRouter
from ai_hub.llm.ollama_client import LLMServiceError, OllamaClient
from ai_hub.schemas.coding_delegation import CodingDelegationPlan
from ai_hub.schemas.coding_actions import (
    CodingAction,
    CodingActionBatch,
    CodingActionResult,
    CreateFileAction,
    DeletePathAction,
    ListFilesAction,
    MakeDirectoryAction,
    ReadFileAction,
    RequestExecutionAction,
)
from ai_hub.tools import code_runner, file_tools


logger = logging.getLogger(__name__)
setup_logging()

SUPPORTED_FILE_EXTENSIONS = {"py", "txt", "md", "json", "yaml", "yml", "toml", "ini", "csv"}
SUPPORTED_SPECIAL_FILENAMES = {"README.md", ".gitignore", "requirements.txt"}
IGNORED_DIRECTORY_TOKENS = {
    "dem",
    "den",
    "deinem",
    "deinen",
    "deiner",
    "dein",
    "einem",
    "einen",
    "einer",
    "mir",
    "workspace",
    "python-workspace",
    "projektroot",
    "projektwurzel",
    "root",
    "mit",
    "wieder",
}


class CodingAgent:
    role = "coding"

    def __init__(
        self,
        language_policy: LanguagePolicy | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        self.language_policy = language_policy or LanguagePolicy()
        self.client = client or OllamaClient()
        self.model = ModelRouter.get_model_for_role("coding")
        self.system_prompt = self._load_system_prompt()

    def handle_task(
        self,
        thread_id: str,
        user_task: str,
        history: list[dict],
        internal_task: str | None = None,
        structured_plan: CodingDelegationPlan | CodingActionBatch | None = None,
    ) -> dict:
        language_context = self.language_policy.build_context(user_task)
        workspace = file_tools.ensure_thread_workspace(thread_id)
        internal_task = internal_task or language_context.internal_message
        blocked_path = self._detect_blocked_path_request(user_task)
        if blocked_path:
            log_event(logger, "coding_blocked_path", thread_id=thread_id, path=blocked_path)
            blocked_result = CodingActionResult(
                action_type="path_guard",
                target=blocked_path,
                status="blocked",
                message=self.language_policy.user_text(
                    language_context.user_language,
                    f"Blockiert: `{blocked_path}` ist außerhalb des erlaubten Coding-Workspaces oder berührt einen sensiblen Bereich.",
                    f"Blocked: `{blocked_path}` is outside the allowed coding workspace or touches a sensitive area.",
                ),
                details={
                    "code": "workspace_path_request_blocked",
                    "path": blocked_path,
                },
            )
            return self._finalize_response(
                thread_id=thread_id,
                workspace=str(workspace),
                internal_task=internal_task,
                executed=[],
                blocked=[blocked_result],
                approval_request=None,
                user_language=language_context.user_language,
            )
        try:
            action_batch, structured_plan_used, fallback_to_heuristic, plan_validation_success = self._resolve_action_batch(
                user_task,
                internal_task,
                structured_plan,
            )
            action_batch = self._inject_contract_reads(
                thread_id=thread_id,
                action_batch=action_batch,
                internal_task=internal_task,
            )
        except Exception as exc:
            log_event(logger, "coding_error", thread_id=thread_id, model=self.model, error=str(exc))
            return self._error_result(thread_id, internal_task, language_context.user_language, exc)
        action_types = [action.action_type for action in action_batch.actions]

        log_event(
            logger,
            "coding_action_batch",
            thread_id=thread_id,
            parsed_action_count=len(action_batch.actions),
            action_types=",".join(action_types),
            blocked_path=blocked_path,
            structured_plan_used=structured_plan_used,
            fallback_to_heuristic=fallback_to_heuristic,
            structured_action_count=len(action_batch.actions) if structured_plan_used else 0,
            plan_validation_success=plan_validation_success,
        )

        execution_result = self._execute_action_batch(
            thread_id=thread_id,
            action_batch=action_batch,
            user_language=language_context.user_language,
        )
        return self._finalize_response(
            thread_id=thread_id,
            workspace=str(workspace),
            internal_task=internal_task,
            executed=execution_result["executed"],
            blocked=execution_result["blocked"],
            approval_request=execution_result["approval_request"],
            user_language=language_context.user_language,
        )

    def _intentionally_empty_file(self, path: str) -> bool:
        normalized = str(path or "").strip().lower()
        if not normalized:
            return False
        filename = normalized.rsplit("/", 1)[-1]
        return filename in {"__init__.py", ".gitkeep", ".keep"}

    def _validate_structured_action_batch(self, action_batch: CodingActionBatch) -> None:
        invalid_paths: list[str] = []
        for action in action_batch.actions:
            if not isinstance(action, CreateFileAction):
                continue
            if self._intentionally_empty_file(action.path):
                continue
            if str(action.content or "").strip():
                continue
            invalid_paths.append(str(action.path))
        if invalid_paths:
            raise ValueError(
                "Structured coding plan contains placeholder-only empty file writes for: "
                + ", ".join(invalid_paths[:4])
            )

    def _execute_action_batch(
        self,
        thread_id: str,
        action_batch: CodingActionBatch,
        user_language: str,
    ) -> dict:
        executed: list[CodingActionResult] = []
        blocked: list[CodingActionResult] = []
        approval_request: dict | None = None
        execution_order: list[str] = []

        for index, action in enumerate(action_batch.actions):
            execution_order.append(action.action_type)
            if isinstance(action, ReadFileAction):
                skipped_read = self._maybe_skip_missing_read(
                    thread_id=thread_id,
                    action=action,
                    user_language=user_language,
                    executed=executed,
                    remaining_actions=action_batch.actions[index + 1:],
                )
                if skipped_read is not None:
                    result = skipped_read
                else:
                    result = self._execute_action(thread_id, action, user_language)
            else:
                result = self._execute_action(thread_id, action, user_language)
            log_event(
                logger,
                "coding_action_result",
                thread_id=thread_id,
                action_type=result.action_type,
                target=result.target,
                status=result.status,
            )
            if result.status == "blocked":
                blocked.append(result)
                break
            executed.append(result)
            if result.action_type == "request_execution":
                approval_request = result.details.get("approval_request")

        log_event(
            logger,
            "coding_action_execution_order",
            thread_id=thread_id,
            action_execution_order=",".join(execution_order),
        )
        return {
            "executed": executed,
            "blocked": blocked,
            "approval_request": approval_request,
        }

    def _maybe_skip_missing_read(
        self,
        *,
        thread_id: str,
        action: ReadFileAction,
        user_language: str,
        executed: list[CodingActionResult],
        remaining_actions: list[CodingAction],
    ) -> CodingActionResult | None:
        try:
            target_path = file_tools.resolve_workspace_path(thread_id, action.path)
        except file_tools.WorkspaceSecurityError:
            return None

        if target_path.exists():
            return None

        batch_contains_mutation = any(
            item.action_type in {"make_directory", "create_file", "delete_path"}
            for item in remaining_actions
        ) or any(
            result.action_type in {"make_directory", "create_file", "delete_path"}
            for result in executed
        )
        if not batch_contains_mutation:
            return None

        log_event(
            logger,
            "coding_read_deferred",
            thread_id=thread_id,
            path=action.path,
            reason="workspace_file_not_found_deferred",
        )
        return CodingActionResult(
            action_type=action.action_type,
            target=action.path,
            status="completed",
            message=self.language_policy.user_text(
                user_language,
                f"Lesen übersprungen: `{action.path}` existiert in diesem Schritt noch nicht und wird später integriert.",
                f"Skipped read: `{action.path}` does not exist yet in this step and will be integrated later.",
            ),
            details={
                "path": action.path,
                "skipped": True,
                "code": "workspace_file_not_found_deferred",
            },
        )

    def _execute_action(
        self,
        thread_id: str,
        action: CodingAction,
        user_language: str,
    ) -> CodingActionResult:
        try:
            if isinstance(action, MakeDirectoryAction):
                result = file_tools.make_directory(thread_id, action.path)
                return CodingActionResult(
                    action_type=action.action_type,
                    target=result["path"],
                    status="completed",
                    message=self.language_policy.user_text(
                        user_language,
                        f"Ordner erstellt: `{result['path']}`.",
                        f"Directory created: `{result['path']}`.",
                    ),
                    details=result,
                )

            if isinstance(action, CreateFileAction):
                result = file_tools.write_file(thread_id, action.path, action.content)
                message = self.language_policy.user_text(
                    user_language,
                    f"Datei erstellt: `{result['path']}`.",
                    f"File created: `{result['path']}`.",
                )
                if action.content:
                    message = (
                        f"{message}\n"
                        + self.language_policy.user_text(
                            user_language,
                            "Inhalt geschrieben.",
                            "Content written.",
                        )
                    )
                return CodingActionResult(
                    action_type=action.action_type,
                    target=result["path"],
                    status="completed",
                    message=message,
                    details={
                        **result,
                        "content_inferred": action.content_inferred,
                    },
                )

            if isinstance(action, ReadFileAction):
                content = file_tools.read_file(thread_id, action.path)
                return CodingActionResult(
                    action_type=action.action_type,
                    target=action.path,
                    status="completed",
                    message=self.language_policy.user_text(
                        user_language,
                        f"Inhalt von `{action.path}`:\n{content}",
                        f"Contents of `{action.path}`:\n{content}",
                    ),
                    details={"path": action.path},
                )

            if isinstance(action, DeletePathAction):
                result = file_tools.delete_path(thread_id, action.path)
                return CodingActionResult(
                    action_type=action.action_type,
                    target=result["path"],
                    status="completed",
                    message=self.language_policy.user_text(
                        user_language,
                        f"{'Datei' if result['kind'] == 'file' else 'Ordner'} gelöscht: `{result['path']}`.",
                        f"{'File' if result['kind'] == 'file' else 'Directory'} deleted: `{result['path']}`.",
                    ),
                    details=result,
                )

            if isinstance(action, ListFilesAction):
                items = file_tools.list_files(thread_id, action.path)
                if items:
                    preview = ", ".join(item["path"] for item in items[:8])
                    message = self.language_policy.user_text(
                        user_language,
                        f"Workspace-Inhalt: {preview}",
                        f"Workspace contents: {preview}",
                    )
                else:
                    message = self.language_policy.user_text(
                        user_language,
                        "Workspace ist aktuell leer.",
                        "The workspace is currently empty.",
                    )
                return CodingActionResult(
                    action_type=action.action_type,
                    target=action.path,
                    status="completed",
                    message=message,
                    details={"items": items, "path": action.path},
                )

            if isinstance(action, RequestExecutionAction):
                approval_request = self._build_execution_request(thread_id, action, user_language)
                return CodingActionResult(
                    action_type=action.action_type,
                    target=action.target,
                    status="approval_required",
                    message=approval_request["user_message"],
                    details={"approval_request": approval_request},
                )
        except file_tools.WorkspaceSecurityError as exc:
            log_event(logger, "coding_workspace_blocked", thread_id=thread_id, code=exc.code, message=exc.user_message)
            return CodingActionResult(
                action_type=action.action_type,
                target=getattr(action, "path", getattr(action, "target", None)),
                status="blocked",
                message=self.language_policy.user_text(
                    user_language,
                    f"Blockiert: {exc.user_message}",
                    f"Blocked: {exc.user_message}",
                ),
                details={"code": exc.code, "message": exc.user_message},
            )
        except code_runner.ExecutionPolicyError as exc:
            log_event(logger, "coding_execution_blocked", thread_id=thread_id, code=exc.code, message=exc.user_message)
            return CodingActionResult(
                action_type=action.action_type,
                target=getattr(action, "path", getattr(action, "target", None)),
                status="blocked",
                message=self.language_policy.user_text(
                    user_language,
                    f"Blockiert: {exc.user_message}",
                    f"Blocked: {exc.user_message}",
                ),
                details={"code": exc.code, "message": exc.user_message},
            )
        except FileNotFoundError as exc:
            log_event(logger, "coding_workspace_missing", thread_id=thread_id, path=exc.args[0])
            return CodingActionResult(
                action_type=action.action_type,
                target=exc.args[0],
                status="blocked",
                message=self.language_policy.user_text(
                    user_language,
                    f"Nicht gefunden: `{exc.args[0]}` existiert im Workspace noch nicht.",
                    f"Not found: `{exc.args[0]}` does not exist in the workspace yet.",
                ),
                details={"code": "workspace_file_not_found"},
            )
        raise ValueError(f"Unsupported coding action: {action}")

    def _finalize_response(
        self,
        thread_id: str,
        workspace: str,
        internal_task: str,
        executed: list[CodingActionResult],
        blocked: list[CodingActionResult],
        approval_request: dict | None,
        user_language: str,
    ) -> dict:
        user_messages = [result.message for result in executed]
        user_messages.extend(result.message for result in blocked)
        actions_executed = [result.model_dump() for result in executed]
        actions_blocked = [result.model_dump() for result in blocked]
        approval_request_created = approval_request is not None
        self_check = self._build_self_check(
            thread_id=thread_id,
            executed=executed,
            blocked=blocked,
            approval_request=approval_request,
            internal_task=internal_task,
        )
        internal_summary = self._build_internal_summary(executed, blocked, approval_request_created, self_check)

        if blocked:
            status = "blocked"
        elif approval_request and any(result.action_type != "request_execution" for result in executed):
            status = "completed_with_approval"
        elif approval_request:
            status = "approval_required"
        else:
            status = "completed"

        if not user_messages:
            user_messages = [
                self.language_policy.user_text(
                    user_language,
                    "Keine direkte Workspace-Aktion erkannt. Ich kann Dateien lesen, schreiben, Ordner anlegen oder eine Ausführung zur Freigabe vorbereiten.",
                    "No direct workspace action detected. I can read files, write files, create directories, or prepare an execution request.",
                )
            ]

        return {
            "reply": "\n".join(user_messages),
            "user_reply": "\n".join(user_messages),
            "status": status,
            "approval_request": approval_request,
            "approval_request_created": approval_request_created,
            "actions_executed": actions_executed,
            "actions_blocked": actions_blocked,
            "tool_results": self._tool_results_from_action_results(executed, blocked),
            "workspace": workspace,
            "internal_summary": internal_summary,
            "internal_payload": self._build_internal_payload(internal_task, thread_id, self_check=self_check),
        }

    def _error_result(self, thread_id: str, internal_task: str, user_language: str, exc: Exception) -> dict:
        code = getattr(exc, "code", "coding_llm_error")
        status_code = getattr(exc, "status_code", None)
        details = str(exc).strip() or code
        if user_language == "de":
            reply = (
                "Coding-Agent Fehler:\n"
                f"- Modell: `{self.model}`\n"
                f"- Fehlercode: `{code}`\n"
                f"- Details: {details}"
            )
            if status_code is not None:
                reply += f"\n- HTTP-Status: `{status_code}`"
        else:
            reply = (
                "Coding agent error:\n"
                f"- Model: `{self.model}`\n"
                f"- Error code: `{code}`\n"
                f"- Details: {details}"
            )
            if status_code is not None:
                reply += f"\n- HTTP status: `{status_code}`"
        return {
            "reply": reply,
            "user_reply": reply,
            "status": "error",
            "approval_request": None,
            "approval_request_created": False,
            "actions_executed": [],
            "actions_blocked": [],
            "tool_results": [],
            "workspace": str(file_tools.ensure_thread_workspace(thread_id)),
            "internal_summary": f"Coding agent failed with {code}.",
            "internal_payload": {
                "language": "en",
                "thread_id": thread_id,
                "task": internal_task,
                "policy": (
                    "Use English for internal routing, worker handoffs, prompts, and tool instructions. "
                    "Keep user-facing communication in the user's language."
                ),
                "source": "error",
                "model": self.model,
                "error": {
                    "code": code,
                    "message": details,
                    "status_code": status_code,
                    "exception_type": type(exc).__name__,
                    "is_llm_error": isinstance(exc, LLMServiceError),
                },
            },
        }

    def _tool_results_from_action_results(
        self,
        executed: list[CodingActionResult],
        blocked: list[CodingActionResult],
    ) -> list[dict]:
        tool_results: list[dict] = []
        for result in executed:
            details = dict(result.details)
            details["operation"] = result.action_type
            tool_results.append(details)
        for result in blocked:
            details = dict(result.details)
            details["operation"] = "blocked"
            details["action_type"] = result.action_type
            tool_results.append(details)
        return tool_results

    def _build_internal_summary(
        self,
        executed: list[CodingActionResult],
        blocked: list[CodingActionResult],
        approval_request_created: bool,
        self_check: dict,
    ) -> str:
        executed_types = ", ".join(result.action_type for result in executed) or "none"
        blocked_types = ", ".join(result.action_type for result in blocked) or "none"
        approval_state = "yes" if approval_request_created else "no"
        self_check_summary = self_check.get("summary", "No self-check summary.")
        return (
            f"Executed actions: {executed_types}. "
            f"Blocked actions: {blocked_types}. "
            f"Approval request created: {approval_state}. "
            f"Self-check: {self_check_summary}"
        )

    def _build_execution_request(
        self,
        thread_id: str,
        action: RequestExecutionAction,
        user_language: str,
    ) -> dict:
        argv, rationale = self._normalize_execution_target(action.target)

        command = code_runner.build_python_execution_request(
            thread_id=thread_id,
            argv=argv,
            rationale=rationale,
        )
        code_runner.validate_python_execution_request(command)
        log_event(
            logger,
            "coding_execution_request_ready",
            thread_id=thread_id,
            command_preview=command.get("preview"),
        )
        return {
            "tool_name": "request_python_execution",
            "command": command,
            "rationale": rationale,
            "user_message": self.language_policy.user_text(
                user_language,
                f"Freigabe nötig: `{command['preview']}` wurde als Ausführungsantrag vorbereitet.",
                f"Approval required: `{command['preview']}` has been prepared as an execution request.",
            ),
        }

    def _normalize_execution_target(self, target: str) -> tuple[list[str], str]:
        cleaned_target = str(target or "").strip()
        if not cleaned_target:
            raise code_runner.ExecutionPolicyError(
                "Leerer Ausführungspfad ist nicht erlaubt.",
                code="execution_empty_command",
            )

        try:
            tokens = shlex.split(cleaned_target)
        except ValueError:
            tokens = cleaned_target.split()
        if not tokens:
            raise code_runner.ExecutionPolicyError(
                "Leerer Ausführungspfad ist nicht erlaubt.",
                code="execution_empty_command",
            )

        if tokens[0] in {"python", "python3"}:
            tokens = tokens[1:]
        if not tokens:
            raise code_runner.ExecutionPolicyError(
                "Leerer Python-Befehl ist nicht erlaubt.",
                code="execution_empty_command",
            )

        if tokens[0] in {"pytest", "unittest"}:
            module = tokens[0]
            argv = ["-m", module, *tokens[1:]]
            rationale = "The coding agent wants to run an allowed Python test command inside the sandbox workspace."
            return argv, rationale

        if tokens[0] == "-m" and len(tokens) >= 2 and tokens[1] in {"pytest", "unittest"}:
            rationale = "The coding agent wants to run an allowed Python test command inside the sandbox workspace."
            return tokens, rationale

        normalized_test_file = self._normalize_workspace_test_file_execution(tokens)
        if normalized_test_file is not None:
            rationale = "The coding agent wants to run a workspace test file through pytest inside the sandbox workspace."
            return normalized_test_file, rationale

        rationale = "The coding agent wants to execute a Python file inside the sandbox workspace."
        return tokens, rationale

    def _normalize_workspace_test_file_execution(self, tokens: list[str]) -> list[str] | None:
        if not tokens:
            return None

        candidate = str(tokens[0]).strip()
        if not candidate.endswith(".py"):
            return None

        normalized = candidate.replace("\\", "/")
        if not normalized.startswith("tests/"):
            return None

        extra_args = list(tokens[1:])
        if "-v" not in extra_args:
            extra_args.append("-v")
        return ["-m", "pytest", candidate, *extra_args]

    def _build_internal_payload(self, internal_task: str, thread_id: str, self_check: dict | None = None) -> dict:
        return {
            "language": "en",
            "thread_id": thread_id,
            "task": internal_task,
            "policy": (
                "Use English for internal routing, worker handoffs, prompts, and tool instructions. "
                "Keep user-facing communication in the user's language."
            ),
            "self_check": self_check or {},
        }

    def _extract_contract_paths(self, internal_task: str) -> dict[str, list[str]]:
        relevant_paths = self._extract_contract_path_line(internal_task, "Relevant existing paths:")
        validation_paths = self._extract_contract_path_line(internal_task, "Relevant validation or entry-point paths:")
        return {
            "relevant_paths": relevant_paths,
            "validation_paths": validation_paths,
        }

    def _extract_contract_path_line(self, internal_task: str, label: str) -> list[str]:
        pattern = re.escape(label) + r"\s*(.+)"
        match = re.search(pattern, internal_task, flags=re.IGNORECASE)
        if not match:
            return []
        raw_value = match.group(1).strip()
        if not raw_value or raw_value.lower() == "none":
            return []
        paths: list[str] = []
        for item in raw_value.split(","):
            candidate = self._normalize_candidate_path(item)
            if candidate and candidate.lower() != "none" and candidate not in paths:
                paths.append(candidate)
        return paths[:8]

    def _inject_contract_reads(
        self,
        *,
        thread_id: str,
        action_batch: CodingActionBatch,
        internal_task: str,
    ) -> CodingActionBatch:
        actions = list(action_batch.actions)
        if not any(action.action_type in {"make_directory", "create_file", "delete_path"} for action in actions):
            return action_batch

        contract_paths = self._extract_contract_paths(internal_task)
        requested_paths = contract_paths["relevant_paths"] + contract_paths["validation_paths"]
        if not requested_paths:
            return action_batch

        existing_paths: list[str] = []
        for path in requested_paths:
            try:
                resolved = file_tools.resolve_workspace_path(thread_id, path)
            except file_tools.WorkspaceSecurityError:
                continue
            if resolved.exists() and path not in existing_paths:
                existing_paths.append(path)

        if not existing_paths:
            return action_batch

        already_read = {
            str(action.path)
            for action in actions
            if isinstance(action, ReadFileAction)
        }
        injected_reads = [
            ReadFileAction(path=path)
            for path in existing_paths
            if path not in already_read
        ]
        if not injected_reads:
            return action_batch
        return CodingActionBatch(actions=[*injected_reads, *actions])

    def _build_self_check(
        self,
        thread_id: str,
        executed: list[CodingActionResult],
        blocked: list[CodingActionResult],
        approval_request: dict | None,
        internal_task: str,
    ) -> dict:
        contract_paths = self._extract_contract_paths(internal_task)
        contract_relevant_paths = contract_paths["relevant_paths"]
        contract_validation_paths = contract_paths["validation_paths"]
        inspected_files: list[dict] = []
        touched_paths: list[str] = []
        for result in executed:
            target = result.target
            if not target:
                continue
            touched_paths.append(target)
            if result.action_type not in {"create_file", "read_file"}:
                continue
            try:
                content = file_tools.read_file(thread_id, target)
            except Exception:
                continue
            inspected_files.append(
                {
                    "path": target,
                    "preview": content[:240],
                    "bytes": len(content.encode("utf-8")),
                }
            )

        inspected_paths = [str(item.get("path", "")).strip() for item in inspected_files if str(item.get("path", "")).strip()]
        contract_paths_inspected = [path for path in contract_relevant_paths if path in inspected_paths]
        validation_paths_inspected = [path for path in contract_validation_paths if path in inspected_paths]

        follow_up: list[str] = []
        if blocked:
            follow_up.append("Resolve the blocked step before expanding the implementation scope.")
        if approval_request is not None:
            follow_up.append("Runtime validation is still pending user approval.")
        if contract_relevant_paths and not contract_paths_inspected:
            follow_up.append("Relevant files from the manager step contract were not re-read during this step.")
        if contract_validation_paths and not validation_paths_inspected:
            follow_up.append("Validation or entry-point files from the manager step contract were not inspected during this step.")

        created_python_files = [
            result.target
            for result in executed
            if result.action_type == "create_file" and str(result.target or "").endswith(".py")
        ]
        if created_python_files and approval_request is None:
            follow_up.append("No runtime validation has been requested yet for the changed Python files.")
        empty_created_files = [
            item["path"]
            for item in inspected_files
            if item["path"] in created_python_files and int(item.get("bytes") or 0) == 0
        ]
        if empty_created_files:
            follow_up.append(
                "The created Python files are still empty, so the requested implementation is not present yet."
            )
        if not executed and not blocked:
            follow_up.append("No concrete workspace mutation happened in this step.")

        summary_parts = []
        if inspected_files:
            preview_paths = ", ".join(item["path"] for item in inspected_files[:3])
            summary_parts.append(f"Inspected files after execution: {preview_paths}.")
        else:
            summary_parts.append("No file contents were re-read after execution.")
        if contract_paths_inspected:
            summary_parts.append("Contract-aligned reads: " + ", ".join(contract_paths_inspected[:4]) + ".")
        if validation_paths_inspected:
            summary_parts.append("Validation-context reads: " + ", ".join(validation_paths_inspected[:4]) + ".")
        if follow_up:
            summary_parts.append("Follow-up: " + " ".join(follow_up[:3]))
        else:
            summary_parts.append("No immediate follow-up risk was detected from the executed action batch.")

        return {
            "inspected_files": inspected_files,
            "touched_paths": touched_paths,
            "contract_relevant_paths": contract_relevant_paths,
            "contract_validation_paths": contract_validation_paths,
            "contract_paths_inspected": contract_paths_inspected,
            "validation_paths_inspected": validation_paths_inspected,
            "follow_up": follow_up,
            "summary": " ".join(summary_parts).strip(),
        }

    def build_action_batch_from_text(self, user_task: str) -> CodingActionBatch:
        return self._build_action_batch(user_task)

    def build_action_batch_from_llm(
        self,
        user_task: str,
        internal_task: str | None = None,
    ) -> CodingActionBatch | None:
        worker_task = internal_task or user_task
        prompt = self._build_llm_prompt(user_task=user_task, internal_task=worker_task)
        log_event(
            logger,
            "coding_ollama_request",
            model=self.model,
            prompt_chars=len(prompt),
            user_task_chars=len(user_task),
            internal_task_chars=len(worker_task),
            internal_task_preview=worker_task[:160],
        )
        if CODING_DEBUG_LOG_PROMPTS:
            log_text_block(
                logger,
                "coding_prompt_body",
                prompt,
                max_chars=CODING_DEBUG_LOG_MAX_CHARS,
                model=self.model,
            )
        response = self.client.generate(model=self.model, prompt=prompt, temperature=0.1)
        log_event(
            logger,
            "coding_generate_completed",
            model=self.model,
            raw_response_chars=len(response),
        )
        if CODING_DEBUG_LOG_PROMPTS:
            log_text_block(
                logger,
                "coding_raw_response",
                response,
                max_chars=CODING_DEBUG_LOG_MAX_CHARS,
                model=self.model,
            )
        return self._parse_llm_action_batch(response)

    def _resolve_action_batch(
        self,
        user_task: str,
        internal_task: str,
        structured_plan: CodingDelegationPlan | CodingActionBatch | None,
    ) -> tuple[CodingActionBatch, bool, bool, bool]:
        if isinstance(structured_plan, CodingDelegationPlan):
            batch = structured_plan.actions
            if batch.actions:
                self._validate_structured_action_batch(batch)
                return batch, True, False, True
            raise ValueError("Structured coding plan is empty.")
        if isinstance(structured_plan, CodingActionBatch):
            if structured_plan.actions:
                self._validate_structured_action_batch(structured_plan)
                return structured_plan, True, False, True
            raise ValueError("Structured coding action batch is empty.")
        llm_batch = self.build_action_batch_from_llm(user_task, internal_task=internal_task)
        if llm_batch is None or not llm_batch.actions:
            raise ValueError("Coding agent returned no executable actions.")
        return llm_batch, True, False, True

    def _load_system_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "coding_agent.txt"
        return prompt_path.read_text(encoding="utf-8").strip()

    def _build_llm_prompt(self, user_task: str, internal_task: str) -> str:
        return f"""
{self.system_prompt}

You are working inside a local multi-agent backend with strict server-side safety controls.
Translate the request into a conservative structured coding action batch.
Only use supported action types and only return actions that fit the user's request.
If the internal task already defines a bounded implementation step, you may write meaningful starter code and tests that fit that step.
Do not return placeholder-only files when the request clearly expects real implementation progress.
Only leave file content empty when the file is intentionally empty, such as a package marker file.
Only use `read_file` for files that already exist or that the user explicitly asked to inspect.
Do not plan reads for future integration files that have not been created yet.
Never return absolute paths or parent-directory escapes.
If no safe coding action is clear, return an empty action list.

Supported action schema examples:
{{
  "actions": [
    {{"action_type": "make_directory", "path": "demo"}},
    {{"action_type": "create_file", "path": "demo/notes.txt", "content": "", "content_inferred": false}},
    {{"action_type": "read_file", "path": "demo/notes.txt"}},
    {{"action_type": "delete_path", "path": "demo/notes.txt"}},
    {{"action_type": "list_files", "path": "."}},
    {{"action_type": "request_execution", "target": "src/main.py"}}
  ]
}}

Original user task:
{user_task}

Internal worker task:
{internal_task}

Return JSON only with this shape:
{{
  "actions": [...]
}}
""".strip()

    def _parse_llm_action_batch(self, response: str) -> CodingActionBatch:
        json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not json_match:
            raise ValueError("Coding agent did not return JSON.")
        payload = json.loads(json_match.group(0))
        if "actions" in payload and isinstance(payload["actions"], list):
            return CodingActionBatch.model_validate(payload)
        if "actions" in payload and isinstance(payload["actions"], dict):
            return CodingActionBatch.model_validate(payload["actions"])
        raise ValueError("Coding agent JSON does not contain a valid action batch.")

    def _build_action_batch(self, user_task: str) -> CodingActionBatch:
        directory = self._clean_directory_value(self._extract_directory_request(user_task))
        file_path = self._extract_file_path(user_task)
        delete_target = self._clean_directory_value(self._extract_delete_target(user_task))
        read_target = self._extract_read_target(user_task)
        should_request_execution = self._should_request_execution(user_task)
        should_list_workspace = self._should_list_workspace(user_task)
        content = ""
        content_inferred = False

        if file_path:
            content = self._infer_file_content(user_task, file_path.split("/")[-1])
            content_inferred = bool(content)

        if not file_path and self._wants_python_file(user_task):
            base_dir = directory or self._clean_directory_value(self._extract_target_directory_hint(user_task)) or "src"
            file_path = f"{base_dir.rstrip('/')}/main.py"
            content = self._infer_file_content(user_task, "main.py")
            content_inferred = bool(content)

        if file_path and directory and "/" not in file_path:
            file_path = f"{directory.rstrip('/')}/{file_path}"
        elif file_path and not directory:
            hinted_dir = self._clean_directory_value(self._extract_target_directory_hint(user_task))
            if hinted_dir and "/" not in file_path:
                file_path = f"{hinted_dir.rstrip('/')}/{file_path}"

        actions: list[CodingAction] = []
        seen_directories: set[str] = set()

        if directory and not delete_target:
            self._append_directory_action(actions, seen_directories, directory)
        if file_path and self._should_create_file(user_task, file_path) and not delete_target:
            parent = file_path.rsplit("/", 1)[0] if "/" in file_path else ""
            if parent:
                self._append_directory_action(actions, seen_directories, parent)
            actions.append(
                CreateFileAction(
                    path=file_path,
                    content=content,
                    content_inferred=content_inferred,
                )
            )
        if read_target:
            actions.append(ReadFileAction(path=read_target))
        if delete_target:
            actions.append(DeletePathAction(path=delete_target))
        if should_list_workspace and not actions:
            actions.append(ListFilesAction())
        if should_request_execution:
            execution_target = self._extract_execution_target(user_task, file_path)
            if execution_target:
                actions.append(RequestExecutionAction(target=execution_target))

        return CodingActionBatch(actions=actions)

    def _append_directory_action(
        self,
        actions: list[CodingAction],
        seen_directories: set[str],
        directory: str,
    ) -> None:
        cleaned_directory = self._clean_directory_value(directory)
        if not cleaned_directory or cleaned_directory in seen_directories:
            return
        seen_directories.add(cleaned_directory)
        actions.append(MakeDirectoryAction(path=cleaned_directory))

    def _detect_blocked_path_request(self, user_task: str) -> str | None:
        candidates = re.findall(
            r"(?:(?<=^)|(?<=[\s(]))(~[^\s,;]+|(?:\.\./)+[^\s,;]*|/[^\s,;]+)",
            user_task,
        )
        sensitive_tokens = [".env", "secrets", ".ssh", ".git", "/etc/", "/var/", "/home/"]

        for candidate in candidates:
            if candidate.startswith("/") or candidate.startswith("~") or ".." in candidate:
                return candidate.rstrip(".,:;!?")

        lowered = user_task.lower()
        for token in sensitive_tokens:
            if token in lowered:
                return token
        return None

    def _infer_file_content(self, user_task: str, filename: str) -> str:
        content_match = re.search(
            r"(?:mit dem inhalt|inhalt|with content|content)\s+(.+?)(?:,?\s+(?:und\s+beantrage|and\s+(?:request|run|execute)).*|$)",
            user_task,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if content_match:
            return content_match.group(1)

        lowered = user_task.lower()
        if filename.endswith(".py") and "hello world" in lowered:
            return 'print("Hello World")\n'
        return ""

    def _extract_directory_request(self, user_task: str) -> str | None:
        patterns = [
            r"(?:unterordner|ordner|verzeichnis)\s+namens\s+([a-zA-Z0-9_./-]+)",
            r"(?:unterordner|ordner|verzeichnis)\s+([a-zA-Z0-9_./-]+)",
            r"(?:in|im|unter)\s+([a-zA-Z0-9_./-]+)\s+ordner",
            r"(?:in|im|unter)\s+([a-zA-Z0-9_./-]+)",
            r"(?:subfolder|folder|directory)\s+(?:named\s+)?([a-zA-Z0-9_./-]+)",
            r"(?:in|under)\s+([a-zA-Z0-9_./-]+)\s+(?:folder|directory)",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_task, flags=re.IGNORECASE)
            if match:
                candidate = self._normalize_candidate_path(match.group(1))
                lowered = candidate.lower()
                if lowered not in IGNORED_DIRECTORY_TOKENS and "workspace" not in lowered:
                    return candidate
        return None

    def _extract_target_directory_hint(self, user_task: str) -> str | None:
        match = re.search(
            r"(?:in|im|unter)\s+([a-zA-Z0-9_./-]+)\s*(?:ordner|verzeichnis)?",
            user_task,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        candidate = self._normalize_candidate_path(match.group(1))
        lowered = candidate.lower()
        if "." in candidate.split("/")[-1] or lowered in IGNORED_DIRECTORY_TOKENS or "workspace" in lowered:
            return None
        return candidate

    def _extract_file_path(self, user_task: str) -> str | None:
        match = re.search(
            r"((?:[a-zA-Z0-9_.-]+/)*(?:[a-zA-Z0-9_.-]+\.(?:py|txt|md|json|yaml|yml|toml|ini|csv)|README\.md|requirements\.txt|\.gitignore))",
            user_task,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return self._normalize_candidate_path(match.group(1))

    def _extract_delete_target(self, user_task: str) -> str | None:
        lowered = user_task.lower()
        if not any(token in lowered for token in ("lösche", "loesche", "delete", "entferne")):
            return None
        path = self._extract_file_path(user_task)
        if path:
            return path
        return self._extract_directory_request(user_task)

    def _extract_read_target(self, user_task: str) -> str | None:
        match = re.search(
            r"(?:lies|öffne|oeffne|zeige|read|open|show).*?([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)",
            user_task,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1).strip().strip("`'\"")

    def _should_create_file(self, user_task: str, file_path: str | None) -> bool:
        lowered = user_task.lower()
        if file_path is None:
            return self._wants_python_file(user_task)
        if any(token in lowered for token in ("lösche", "loesche", "delete", "entferne", "lies", "öffne", "zeige")):
            return any(token in lowered for token in ("erstelle", "schreibe", "lege an", "create", "write"))
        return any(phrase in lowered for phrase in ("erstelle", "schreibe", "lege an", "create", "write", "darin"))

    def _should_list_workspace(self, user_task: str) -> bool:
        lowered = user_task.lower()
        return "list" in lowered or "datei" in lowered or "workspace" in lowered

    def _wants_python_file(self, user_task: str) -> bool:
        lowered = user_task.lower()
        return "python-datei" in lowered or "python datei" in lowered or "python file" in lowered

    def _should_request_execution(self, user_task: str) -> bool:
        lowered = user_task.lower()
        return any(
            phrase in lowered
            for phrase in (
                "beantrage anschließend die ausführung",
                "beantrage die ausführung",
                "anschließend ausführen",
                "ausführung beantragen",
                "bitte ausführen",
                "request the execution",
                "request execution",
                "then run",
                "afterward run",
                "execute it",
                "run it",
                "starte ",
                "führe ",
                "ausführen",
                "pytest",
                "debug",
            )
        )

    def _extract_execution_target(self, user_task: str, created_file_path: str | None) -> str | None:
        if created_file_path:
            return created_file_path
        path = self._extract_file_path(user_task)
        if path:
            return path
        if "pytest" in user_task.lower():
            return "pytest"
        return None

    def _normalize_candidate_path(self, raw_value: str) -> str:
        value = raw_value.strip().strip("`'\"")
        value = value.rstrip(".,:;!?")
        value = value.rstrip("/")
        return value

    def _clean_directory_value(self, value: str | None) -> str | None:
        if not value:
            return None
        lowered = value.lower()
        if lowered in IGNORED_DIRECTORY_TOKENS:
            return None
        if any(token in lowered for token in ("workspace", "projektroot", "projektwurzel")):
            return None
        return value
