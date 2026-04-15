import logging
import re

from ai_hub.language_policy import LanguagePolicy
from ai_hub.logging_config import log_event, setup_logging
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

    def __init__(self, language_policy: LanguagePolicy | None = None) -> None:
        self.language_policy = language_policy or LanguagePolicy()

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
        action_batch, structured_plan_used, fallback_to_heuristic, plan_validation_success = self._resolve_action_batch(
            user_task,
            structured_plan,
        )
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

        for action in action_batch.actions:
            execution_order.append(action.action_type)
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
        internal_summary = self._build_internal_summary(executed, blocked, approval_request_created)

        if blocked:
            status = "blocked"
        elif approval_request and executed:
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
            "internal_payload": self._build_internal_payload(internal_task, thread_id),
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
    ) -> str:
        executed_types = ", ".join(result.action_type for result in executed) or "none"
        blocked_types = ", ".join(result.action_type for result in blocked) or "none"
        approval_state = "yes" if approval_request_created else "no"
        return (
            f"Executed actions: {executed_types}. "
            f"Blocked actions: {blocked_types}. "
            f"Approval request created: {approval_state}."
        )

    def _build_execution_request(
        self,
        thread_id: str,
        action: RequestExecutionAction,
        user_language: str,
    ) -> dict:
        lowered_target = action.target.lower()
        pytest_match = lowered_target.endswith(".py") is False and action.target == "pytest"

        if pytest_match:
            argv = ["-m", "pytest"]
            rationale = "The coding agent wants to run an allowed Python test command inside the sandbox workspace."
        else:
            argv = [action.target]
            rationale = "The coding agent wants to execute a Python file inside the sandbox workspace."

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

    def _build_internal_payload(self, internal_task: str, thread_id: str) -> dict:
        return {
            "language": "en",
            "thread_id": thread_id,
            "task": internal_task,
            "policy": (
                "Use English for internal routing, worker handoffs, prompts, and tool instructions. "
                "Keep user-facing communication in the user's language."
            ),
        }

    def build_action_batch_from_text(self, user_task: str) -> CodingActionBatch:
        return self._build_action_batch(user_task)

    def _resolve_action_batch(
        self,
        user_task: str,
        structured_plan: CodingDelegationPlan | CodingActionBatch | None,
    ) -> tuple[CodingActionBatch, bool, bool, bool]:
        if isinstance(structured_plan, CodingDelegationPlan):
            batch = structured_plan.actions
            if batch.actions:
                return batch, True, False, True
            return self._build_action_batch(user_task), False, True, False
        if isinstance(structured_plan, CodingActionBatch):
            if structured_plan.actions:
                return structured_plan, True, False, True
            return self._build_action_batch(user_task), False, True, False
        return self._build_action_batch(user_task), False, True, False

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
            r"(?:mit dem inhalt|inhalt|with content|content)\s+(.+?)(?:,?\s+und\s+beantrage.*|$)",
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
            r"(?:lies|öffne|zeige).*?([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)",
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
