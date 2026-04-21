import hashlib
import logging
import re
from uuid import uuid4
from time import monotonic, sleep

from ai_hub.config import (
    MANAGER_AUTONOMOUS_MAX_DEBUG_REPAIRS,
    MANAGER_AUTONOMOUS_MAX_STEPS,
    POST_CODING_REVIEW_DELAY_SECONDS,
    REVIEWER_DEBUG_LOG_MAX_CHARS,
    REVIEWER_DEBUG_LOG_PROMPTS,
)
from ai_hub.language_policy import LanguagePolicy
from ai_hub.logging_config import (
    get_request_id,
    log_event,
    log_text_block,
    reset_request_id,
    set_request_id,
    setup_logging,
)
from ai_hub.memory.history import format_thread_history
from ai_hub.memory.store import HubStore
from ai_hub.llm.manager_planner import ManagerPlanner
from ai_hub.orchestration.delegation import DelegationService
from ai_hub.schemas.coding_delegation import CodingDelegationPlan
from ai_hub.schemas.manager_plan import ManagerPlan, ProjectOutline
from ai_hub.state import ApprovalStatus, ManagerDecision, RouteDecision
from ai_hub.tools.code_runner import (
    ExecutionPolicyError,
    build_python_package_install_request,
    execute_python_approval,
)
from ai_hub.tools.dependency_tools import analyze_workspace_dependencies
from ai_hub.tools import file_tools
from ai_hub.tools.file_tools import WorkspaceSecurityError
from ai_hub.tools.push_notify import send_notification


logger = logging.getLogger(__name__)
setup_logging()

ROLE_CONFLICT_PATTERNS = (
    r"\bich bin der coding-agent\b",
    r"\bi am the coding agent\b",
    r"\bich bin der research-agent\b",
    r"\bi am the research agent\b",
    r"\bich bin der reviewer\b",
    r"\bi am the reviewer\b",
)

STRATEGIC_USER_PATTERNS = (
    "wie würdest du",
    "how would you",
    "strukturieren",
    "structure",
    "architektur",
    "architecture",
    "plan",
    "planen",
    "konzept",
)

PLAN_REQUEST_PATTERNS = (
    "implementierungsplan",
    "implementation plan",
    "erst planen",
    "first plan",
    "bevor du den coding agenten beauftragst",
    "before you ask the coding agent",
    "roadmap",
    "arbeitspakete",
    "milestones",
    "projektplan",
)

PROJECT_CONTINUE_PATTERNS = (
    "mach weiter",
    "weiter so",
    "leg los",
    "setze um",
    "implementiere",
    "fang an",
    "go ahead",
    "continue",
    "proceed",
    "looks good",
    "sounds good",
    "passt so",
    "gefällt mir",
    "einverstanden",
)

PROJECT_CHANGE_PATTERNS = (
    "change",
    "update",
    "modify",
    "adjust",
    "tweak",
    "switch",
    "replace",
    "rename",
    "remove",
    "add ",
    "instead",
    "different",
    "now make",
    "now change",
    "ändere",
    "aendere",
    "ändere jetzt",
    "passe",
    "anpassen",
    "tausche",
    "ersetze",
    "entferne",
    "füge",
    "fuege",
    "statt",
    "nun",
)

PROJECT_REPAIR_PATTERNS = (
    "fix",
    "repair",
    "repar",
    "debug",
    "retry",
    "nochmal",
    "erneut",
    "weiter",
    "continue",
)

FULL_IMPLEMENTATION_PATTERNS = (
    "let me know when you are done",
    "let me know when the full implementation is done",
    "when the full implementation is done",
    "start the implementation",
    "finish the implementation",
    "implement everything",
    "full implementation",
    "when you are done",
    "tell me when you are done",
    "start implementation",
    "complete the implementation",
    "fertig implementieren",
    "wenn du fertig bist",
    "lass es komplett implementieren",
)

LARGE_CODING_PATTERNS = (
    "projekt",
    "project",
    "app",
    "application",
    "system",
    "feature",
    "workflow",
    "refactor",
    "mehrere dateien",
    "mehrere module",
    "mehrschritt",
)

EXPLICIT_WORKSPACE_ACTION_PATTERNS = (
    r"\berstelle\b.+\b(datei|file|ordner|unterordner|verzeichnis|pfad|workspace|sandbox)\b",
    r"\bschreibe\b.+\b(datei|file|workspace|sandbox)\b",
    r"\blege\b.+\b(an|datei|file|ordner|unterordner|verzeichnis)\b",
    r"\blies\b.+\b(datei|file|workspace|ordner|pfad)\b",
    r"\b(öffne|oeffne|zeige)\b.+\b(datei|file|ordner|pfad)\b",
    r"\b(lösche|loesche|delete|entferne)\b.+\b(datei|file|ordner|unterordner|verzeichnis|pfad)\b",
    r"\bbeantrage\b.+\bausführung\b",
    r"\bausführen\b",
    r"\bpytest\b",
    r"\b[\w./-]+\.(py|txt|md|json|yaml|yml|csv|toml|ini)\b",
)

RAW_CONTENT_REQUEST_PATTERNS = (
    r"\b(lies|read|show|zeige|open|oeffne)\b.+\b(inhalt|contents?|content|datei|file)\b",
    r"\bcat\b.+\b[\w./-]+\b",
)

DEPENDENCY_INSTALL_FORBID_PATTERNS = (
    "do not install any dependencies",
    "do not install dependencies",
    "don't install any dependencies",
    "don't install dependencies",
    "installiere keine abhängigkeiten",
    "installiere keine dependencies",
    "keine abhängigkeiten installieren",
    "keine dependencies installieren",
)

DEPENDENCY_INSTALL_ALLOW_PATTERNS = (
    "you can install dependencies",
    "you can install packages",
    "install dependencies now",
    "install the dependencies now",
    "du kannst abhängigkeiten installieren",
    "du kannst dependencies installieren",
    "installiere jetzt die abhängigkeiten",
)

PYTEST_FORBID_PATTERNS = (
    "do not run pytest",
    "don't run pytest",
    "run no pytest",
    "führe pytest nicht aus",
    "pytest nicht ausführen",
    "kein pytest",
)

PYTEST_ALLOW_PATTERNS = (
    "you can run pytest",
    "run pytest now",
    "führe pytest aus",
    "pytest jetzt",
)

EXECUTION_FORBID_PATTERNS = (
    "do not request execution",
    "don't request execution",
    "do not run the code yet",
    "don't run the code yet",
    "do not execute yet",
    "don't execute yet",
    "keine ausführung",
    "nicht ausführen",
    "nicht starten",
)

EXECUTION_ALLOW_PATTERNS = (
    "you can run it",
    "you can execute it",
    "request execution",
    "beantrage die ausführung",
    "führe es aus",
)

STOP_AFTER_STEP_PATTERNS = (
    "and stop",
    "und stop",
    "und dann stoppen",
    "halte danach an",
    "stop after",
    "after that, report",
    "after that report",
    "report exactly",
    "danach report",
    "danach berichte",
    "berichte danach",
)

DIRECTIVE_SENTENCE_PATTERNS = (
    "do not",
    "don't",
    "only",
    "must",
    "keep",
    "use ",
    "store ",
    "report ",
    "stop",
    "first ",
    "zuerst",
    "nutze",
    "verwende",
    "speichere",
    "halte",
    "berichte",
)


class ManagerWorkflow:
    def __init__(
        self,
        store: HubStore | None = None,
        planner: ManagerPlanner | None = None,
    ) -> None:
        self.store = store or HubStore()
        self.language_policy = LanguagePolicy()
        self.delegation = DelegationService(store=self.store)
        self.planner = planner or ManagerPlanner()

    def handle_chat(self, thread_id: str | None, user_message: str) -> dict:
        token = None
        if get_request_id() == "-":
            token = set_request_id(f"chat-{uuid4().hex[:12]}")
        language_context = self.language_policy.build_context(user_message)
        thread = self.store.ensure_thread(thread_id)
        cleaned_message = language_context.user_message.strip()
        manager_user_language = language_context.user_language
        if not cleaned_message:
            if token is not None:
                reset_request_id(token)
            raise ValueError("Leere Nachrichten sind nicht erlaubt.")
        log_event(
            logger,
            "chat_received",
            thread_id=thread["id"],
            user_message=cleaned_message,
        )

        try:
            self.store.add_message(thread["id"], role="user", content=cleaned_message)
            self.store.rename_thread_from_first_message(thread["id"], cleaned_message)
            manager_user_language = self._preferred_manager_language(
                thread["id"],
                language_context.user_language,
                cleaned_message,
            )
            self._store_manager_constraints(thread["id"], cleaned_message)
            worker_user_message = (
                language_context.internal_message
                if manager_user_language == "en"
                else cleaned_message
            )
            history = self.store.list_messages(thread["id"])
            history_text = self._build_planning_context(thread["id"], history[:-1], limit=6)
            planning_note = self._plan_safely(
                history_text,
                language_context.internal_message,
                manager_user_language,
            )
            if planning_note.get("source") == "error":
                return self._store_error_response(
                    thread_id=thread["id"],
                    reply=self._format_llm_error_reply(
                        user_language=manager_user_language,
                        agent_label="Manager",
                        model=planning_note.get("model") or getattr(self.planner, "model", "unknown"),
                        code=planning_note.get("error_code", "manager_planner_error"),
                        details=planning_note.get("error", "Unbekannter Fehler."),
                        status_code=planning_note.get("status_code"),
                    ),
                    meta={
                        "route": "error",
                        "approval_request_id": None,
                        "planning_note": self._serialize_planning_note(planning_note),
                        "manager_source": "error",
                        "fallback_reason": None,
                        "llm_decision": None,
                        "final_decision": "error",
                        "structured_plan_used": False,
                        "fallback_to_heuristic": False,
                        "structured_action_count": 0,
                        "plan_validation_success": False,
                        "language_policy": {
                            "user_language": manager_user_language,
                            "internal_language": "en",
                        },
                        "internal_payload": {
                            "language": "en",
                            "thread_id": thread["id"],
                            "task": language_context.internal_message,
                            "source": "error",
                            "model": planning_note.get("model") or getattr(self.planner, "model", "unknown"),
                            "error": {
                                "code": planning_note.get("error_code", "manager_planner_error"),
                                "message": planning_note.get("error", "Unbekannter Fehler."),
                                "status_code": planning_note.get("status_code"),
                            },
                        },
                    },
                )
            route = self._resolve_route(thread["id"], cleaned_message, planning_note)
            llm_decision = self._llm_decision(planning_note)
            approval = None
            delegated_result = None
            manager_source = "ollama"
            fallback_reason = None
            coding_structured_plan = None
            structured_plan_used = False
            fallback_to_heuristic = False
            structured_action_count = 0
            plan_validation_success = False

            if route.decision == ManagerDecision.DIRECT:
                reply, manager_source, direct_fallback_reason = self._direct_reply(
                    cleaned_message,
                    manager_user_language,
                    planning_note,
                )
                fallback_reason = direct_fallback_reason
            elif route.decision == ManagerDecision.PLAN:
                project_plan_result = self._execute_project_planning(
                    thread_id=thread["id"],
                    user_message=worker_user_message,
                    history=history,
                    planning_note=planning_note,
                    user_language=manager_user_language,
                )
                delegated_result = project_plan_result
                reply = project_plan_result["reply"]
            else:
                execution = self._execute_worker_route(
                    thread_id=thread["id"],
                    user_message=worker_user_message,
                    history=history,
                    route=route,
                    planning_note=planning_note,
                    fallback_internal_message=language_context.internal_message,
                    user_language=manager_user_language,
                )
                delegated_result = execution["delegated_result"]
                reply = execution["reply"]
                coding_structured_plan = execution["coding_structured_plan"]
                structured_plan_used = execution["structured_plan_used"]
                fallback_to_heuristic = execution["fallback_to_heuristic"]
                plan_validation_success = execution["plan_validation_success"]
                structured_action_count = execution["structured_action_count"]
                approval_payload = delegated_result.get("approval_request")
                if approval_payload:
                    approval = self.store.create_approval_request(
                        thread_id=thread["id"],
                        agent_role=route.decision.value,
                        tool_name=approval_payload["tool_name"],
                        command=approval_payload["command"],
                        rationale=approval_payload["rationale"],
                    )
                    log_event(
                        logger,
                        "approval_created",
                        thread_id=thread["id"],
                        approval_id=approval["id"],
                        agent_role=route.decision.value,
                        tool_name=approval_payload["tool_name"],
                        command_preview=approval_payload["command"].get("preview"),
                    )
                    try:
                        send_notification(
                            title=self.language_policy.user_text(
                                manager_user_language,
                                "AI Hub Freigabe erforderlich",
                                "AI Hub approval required",
                            ),
                            body=self.language_policy.user_text(
                                manager_user_language,
                                f"Thread „{self.store.get_thread(thread['id'])['title']}“ wartet auf deine Bestätigung.",
                                f"Thread “{self.store.get_thread(thread['id'])['title']}” is waiting for your approval.",
                            ),
                            url=f"/?thread={thread['id']}&approval={approval['id']}",
                        )
                    except Exception:
                        pass

            delegated_agent = route.decision.value if route.decision != ManagerDecision.DIRECT else "-"
            log_event(
                logger,
                "manager_decision",
                thread_id=thread["id"],
                source=manager_source,
                llm_decision=llm_decision,
                final_decision=route.decision.value,
                fallback_reason=fallback_reason,
                delegated_agent=delegated_agent,
                approval_created=bool(approval),
                structured_plan_used=structured_plan_used,
                fallback_to_heuristic=fallback_to_heuristic,
                structured_action_count=structured_action_count,
                plan_validation_success=plan_validation_success,
            )

            assistant_message = self.store.add_message(
                thread["id"],
                role="assistant",
                content=reply,
                agent="manager",
                meta={
                    "route": route.decision.value,
                    "approval_request_id": approval["id"] if approval else None,
                    "planning_note": self._serialize_planning_note(planning_note),
                    "manager_source": manager_source,
                    "fallback_reason": fallback_reason,
                    "llm_decision": llm_decision,
                    "final_decision": route.decision.value,
                    "structured_plan_used": structured_plan_used,
                    "fallback_to_heuristic": fallback_to_heuristic,
                    "structured_action_count": structured_action_count,
                    "plan_validation_success": plan_validation_success,
                    "language_policy": {
                        "user_language": manager_user_language,
                        "internal_language": "en",
                    },
                    "internal_payload": delegated_result.get("internal_payload") if delegated_result else None,
                },
            )
            thread = self.store.get_thread(thread["id"])

            return {
                "thread": thread,
                "message": assistant_message,
                "reply": reply,
                "route": route.decision.value,
                "approval_request": approval,
                "messages": self.store.list_messages(thread["id"]),
            }
        except Exception as exc:
            log_event(
                logger,
                "chat_processing_error",
                thread_id=thread["id"],
                error=str(exc),
                exception_type=type(exc).__name__,
            )
            model = getattr(self.planner, "model", "unknown")
            code = getattr(exc, "code", "chat_processing_error")
            status_code = getattr(exc, "status_code", None)
            return self._store_error_response(
                thread_id=thread["id"],
                reply=self._format_llm_error_reply(
                    user_language=language_context.user_language,
                    agent_label="Manager",
                    model=model,
                    code=code,
                    details=str(exc) or code,
                    status_code=status_code,
                ),
                meta={
                    "route": "error",
                    "approval_request_id": None,
                    "planning_note": self._serialize_planning_note(locals().get("planning_note")),
                    "manager_source": "error",
                    "fallback_reason": None,
                    "llm_decision": None,
                    "final_decision": "error",
                    "structured_plan_used": False,
                    "fallback_to_heuristic": False,
                    "structured_action_count": 0,
                    "plan_validation_success": False,
                    "language_policy": {
                        "user_language": manager_user_language,
                        "internal_language": "en",
                    },
                    "internal_payload": {
                        "language": "en",
                        "thread_id": thread["id"],
                        "task": language_context.internal_message,
                        "source": "error",
                        "model": model,
                        "error": {
                            "code": code,
                            "message": str(exc) or code,
                            "status_code": status_code,
                            "exception_type": type(exc).__name__,
                        },
                    },
                },
            )
        finally:
            if token is not None:
                reset_request_id(token)

    def process_enqueued_chat(self, thread_id: str, user_message: str, pending_message_id: int) -> dict:
        token = None
        if get_request_id() == "-":
            token = set_request_id(f"chat-{uuid4().hex[:12]}")
        language_context = self.language_policy.build_context(user_message)
        thread = self.store.get_thread(thread_id)
        if thread is None:
            raise KeyError("Thread nicht gefunden.")
        cleaned_message = language_context.user_message.strip()
        manager_user_language = language_context.user_language
        if not cleaned_message:
            raise ValueError("Leere Nachrichten sind nicht erlaubt.")

        log_event(
            logger,
            "chat_processing_started",
            thread_id=thread_id,
            pending_message_id=pending_message_id,
            user_message=cleaned_message,
        )

        try:
            history = self.store.list_messages(thread_id)
            history_without_pending = history[:-1] if history and history[-1]["id"] == pending_message_id else history
            manager_user_language = self._preferred_manager_language(
                thread_id,
                language_context.user_language,
                cleaned_message,
            )
            self._store_manager_constraints(thread_id, cleaned_message)
            worker_user_message = (
                language_context.internal_message
                if manager_user_language == "en"
                else cleaned_message
            )
            self._update_pending_status(
                pending_message_id=pending_message_id,
                comment=self.language_policy.user_text(
                    manager_user_language,
                    "Anfrage wird eingeordnet",
                    "Understanding the request",
                ),
            )
            history_text = self._build_planning_context(thread_id, history_without_pending[:-1], limit=6)
            planning_note = self._plan_safely(
                history_text,
                language_context.internal_message,
                manager_user_language,
            )
            if planning_note.get("source") == "error":
                return self._finalize_pending_message(
                    thread_id=thread_id,
                    pending_message_id=pending_message_id,
                    reply=self._format_llm_error_reply(
                        user_language=manager_user_language,
                        agent_label="Manager",
                        model=planning_note.get("model") or getattr(self.planner, "model", "unknown"),
                        code=planning_note.get("error_code", "manager_planner_error"),
                        details=planning_note.get("error", "Unbekannter Fehler."),
                        status_code=planning_note.get("status_code"),
                    ),
                    meta={
                        "route": "error",
                        "approval_request_id": None,
                        "planning_note": self._serialize_planning_note(planning_note),
                        "manager_source": "error",
                        "fallback_reason": None,
                        "llm_decision": None,
                        "final_decision": "error",
                        "structured_plan_used": False,
                        "fallback_to_heuristic": False,
                        "structured_action_count": 0,
                        "plan_validation_success": False,
                        "processing": False,
                        "processing_started_at": history[-1]["meta"].get("processing_started_at") if history and history[-1]["id"] == pending_message_id else None,
                        "completed_at": self.store.get_thread(thread_id)["updated_at"] if self.store.get_thread(thread_id) else None,
                        "language_policy": {
                            "user_language": manager_user_language,
                            "internal_language": "en",
                        },
                        "internal_payload": {
                            "language": "en",
                            "thread_id": thread_id,
                            "task": language_context.internal_message,
                            "source": "error",
                            "model": planning_note.get("model") or getattr(self.planner, "model", "unknown"),
                            "error": {
                                "code": planning_note.get("error_code", "manager_planner_error"),
                                "message": planning_note.get("error", "Unbekannter Fehler."),
                                "status_code": planning_note.get("status_code"),
                            },
                        },
                    },
                    route="error",
                    approval=None,
                )

            route = self._resolve_route(thread_id, cleaned_message, planning_note)
            llm_decision = self._llm_decision(planning_note)
            self._update_pending_status(
                pending_message_id=pending_message_id,
                comment=self._processing_comment_for_route(route.decision, manager_user_language),
            )
            approval = None
            delegated_result = None
            manager_source = "ollama"
            fallback_reason = None
            coding_structured_plan = None
            structured_plan_used = False
            fallback_to_heuristic = False
            structured_action_count = 0
            plan_validation_success = False

            if route.decision == ManagerDecision.DIRECT:
                reply, manager_source, direct_fallback_reason = self._direct_reply(
                    cleaned_message,
                    manager_user_language,
                    planning_note,
                )
                fallback_reason = direct_fallback_reason
            elif route.decision == ManagerDecision.PLAN:
                project_plan_result = self._execute_project_planning(
                    thread_id=thread_id,
                    user_message=worker_user_message,
                    history=history_without_pending,
                    planning_note=planning_note,
                    user_language=manager_user_language,
                    pending_message_id=pending_message_id,
                )
                delegated_result = project_plan_result
                reply = project_plan_result["reply"]
            else:
                execution = self._execute_worker_route(
                    thread_id=thread_id,
                    user_message=worker_user_message,
                    history=history_without_pending,
                    route=route,
                    planning_note=planning_note,
                    fallback_internal_message=language_context.internal_message,
                    user_language=manager_user_language,
                    pending_message_id=pending_message_id,
                )
                delegated_result = execution["delegated_result"]
                reply = execution["reply"]
                coding_structured_plan = execution["coding_structured_plan"]
                structured_plan_used = execution["structured_plan_used"]
                fallback_to_heuristic = execution["fallback_to_heuristic"]
                plan_validation_success = execution["plan_validation_success"]
                structured_action_count = execution["structured_action_count"]
                approval_payload = delegated_result.get("approval_request")
                if approval_payload:
                    self._update_pending_status(
                        pending_message_id=pending_message_id,
                        comment=self.language_policy.user_text(
                            manager_user_language,
                            "Freigabe wird vorbereitet",
                            "Preparing the approval request",
                        ),
                    )
                    approval = self.store.create_approval_request(
                        thread_id=thread_id,
                        agent_role=route.decision.value,
                        tool_name=approval_payload["tool_name"],
                        command=approval_payload["command"],
                        rationale=approval_payload["rationale"],
                    )
                    log_event(
                        logger,
                        "approval_created",
                        thread_id=thread_id,
                        approval_id=approval["id"],
                        agent_role=route.decision.value,
                        tool_name=approval_payload["tool_name"],
                        command_preview=approval_payload["command"].get("preview"),
                    )
                    try:
                        send_notification(
                            title=self.language_policy.user_text(
                                manager_user_language,
                                "AI Hub Freigabe erforderlich",
                                "AI Hub approval required",
                            ),
                            body=self.language_policy.user_text(
                                manager_user_language,
                                f"Thread „{self.store.get_thread(thread_id)['title']}“ wartet auf deine Bestätigung.",
                                f"Thread “{self.store.get_thread(thread_id)['title']}” is waiting for your approval.",
                            ),
                            url=f"/?thread={thread_id}&approval={approval['id']}",
                        )
                    except Exception:
                        pass

            delegated_agent = route.decision.value if route.decision != ManagerDecision.DIRECT else "-"
            log_event(
                logger,
                "manager_decision",
                thread_id=thread_id,
                source=manager_source,
                llm_decision=llm_decision,
                final_decision=route.decision.value,
                fallback_reason=fallback_reason,
                delegated_agent=delegated_agent,
                approval_created=bool(approval),
                structured_plan_used=structured_plan_used,
                fallback_to_heuristic=fallback_to_heuristic,
                structured_action_count=structured_action_count,
                plan_validation_success=plan_validation_success,
            )

            log_event(
                logger,
                "chat_processing_finalization_started",
                thread_id=thread_id,
                pending_message_id=pending_message_id,
                final_route=route.decision.value,
                approval_created=bool(approval),
                reply_chars=len(reply),
            )
            return self._finalize_pending_message(
                thread_id=thread_id,
                pending_message_id=pending_message_id,
                reply=reply,
                meta={
                    "route": route.decision.value,
                    "approval_request_id": approval["id"] if approval else None,
                    "planning_note": self._serialize_planning_note(planning_note),
                    "manager_source": manager_source,
                    "fallback_reason": fallback_reason,
                    "llm_decision": llm_decision,
                    "final_decision": route.decision.value,
                    "structured_plan_used": structured_plan_used,
                    "fallback_to_heuristic": fallback_to_heuristic,
                    "structured_action_count": structured_action_count,
                    "plan_validation_success": plan_validation_success,
                    "processing": False,
                    "processing_started_at": history[-1]["meta"].get("processing_started_at") if history and history[-1]["id"] == pending_message_id else None,
                    "language_policy": {
                        "user_language": manager_user_language,
                        "internal_language": "en",
                    },
                    "internal_payload": delegated_result.get("internal_payload") if delegated_result else None,
                },
                route=route.decision.value,
                approval=approval,
            )
        except Exception as exc:
            log_event(
                logger,
                "chat_processing_error",
                thread_id=thread_id,
                pending_message_id=pending_message_id,
                error=str(exc),
                exception_type=type(exc).__name__,
            )
            model = getattr(self.planner, "model", "unknown")
            code = getattr(exc, "code", "chat_processing_error")
            status_code = getattr(exc, "status_code", None)
            return self._finalize_pending_message(
                thread_id=thread_id,
                pending_message_id=pending_message_id,
                reply=self._format_llm_error_reply(
                    user_language=manager_user_language,
                    agent_label="Manager",
                    model=model,
                    code=code,
                    details=str(exc) or code,
                    status_code=status_code,
                ),
                meta={
                    "route": "error",
                    "approval_request_id": None,
                    "planning_note": self._serialize_planning_note(locals().get("planning_note")),
                    "manager_source": "error",
                    "fallback_reason": None,
                    "llm_decision": None,
                    "final_decision": "error",
                    "structured_plan_used": False,
                    "fallback_to_heuristic": False,
                    "structured_action_count": 0,
                    "plan_validation_success": False,
                    "processing": False,
                    "processing_started_at": history[-1]["meta"].get("processing_started_at") if "history" in locals() and history and history[-1]["id"] == pending_message_id else None,
                    "language_policy": {
                        "user_language": manager_user_language,
                        "internal_language": "en",
                    },
                    "internal_payload": {
                        "language": "en",
                        "thread_id": thread_id,
                        "task": language_context.internal_message,
                        "source": "error",
                        "model": model,
                        "error": {
                            "code": code,
                            "message": str(exc) or code,
                            "status_code": status_code,
                            "exception_type": type(exc).__name__,
                        },
                    },
                },
                route="error",
                approval=None,
            )
        finally:
            if token is not None:
                reset_request_id(token)

    def _mark_autonomous_approval_outcome(self, thread_id: str, result: dict) -> dict | None:
        project_state = self._get_project_state(thread_id)
        if not project_state or not project_state.get("autonomous_mode"):
            return None

        pending_step = project_state.get("pending_step")
        completed_steps = list(project_state.get("completed_steps") or [])
        outline = self._project_outline_from_memory(thread_id)
        if not outline:
            return None
        command_kind = result.get("command_kind")
        ordered_steps = self._ordered_project_steps(outline)

        if result.get("returncode") == 0:
            if command_kind == "pip_install":
                remaining_steps = [step for step in ordered_steps if step not in completed_steps]
                ready_to_test = False
                self.store.upsert_artifact(
                    thread_id=thread_id,
                    kind="project_state",
                    title="Project State",
                    summary="Approved dependency installation completed successfully.",
                    content={
                        **project_state,
                        "phase": "implementation",
                        "awaiting_user_feedback": False,
                        "latest_status": "dependencies_installed",
                        "ready_to_test": False,
                        "completed_steps": completed_steps,
                        "pending_step": None,
                        "remaining_steps": remaining_steps,
                        "next_steps": remaining_steps[:3],
                        "last_execution_result": result,
                    },
                )
                return {"ready_to_test": False, "completed_steps": completed_steps}

            if pending_step and pending_step not in completed_steps:
                completed_steps.append(pending_step)
            ready_to_test = len(completed_steps) >= len(ordered_steps)
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="project_state",
                title="Project State",
                summary="Approved execution completed successfully.",
                content={
                    **project_state,
                    "phase": "ready_to_test" if ready_to_test else "implementation",
                    "awaiting_user_feedback": bool(ready_to_test),
                    "latest_status": "executed" if ready_to_test else "completed",
                    "ready_to_test": ready_to_test,
                    "completed_steps": completed_steps,
                    "pending_step": None,
                    "remaining_steps": [step for step in ordered_steps if step not in completed_steps],
                    "next_steps": [step for step in ordered_steps if step not in completed_steps][:3],
                    "last_execution_result": result,
                },
            )
            return {"ready_to_test": ready_to_test, "completed_steps": completed_steps}

        debug_attempts = int(project_state.get("debug_attempts") or 0)
        can_retry_debug = (
            command_kind == "python_execution"
            and pending_step is not None
            and debug_attempts < MANAGER_AUTONOMOUS_MAX_DEBUG_REPAIRS
        )
        if can_retry_debug:
            remaining_steps = [step for step in ordered_steps if step not in completed_steps]
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="project_state",
                title="Project State",
                summary="Approved execution failed and the manager is retrying with a focused repair step.",
                content={
                    **project_state,
                    "phase": "debugging",
                    "awaiting_user_feedback": False,
                    "latest_status": "execution_failed_retrying",
                    "ready_to_test": False,
                    "pending_step": None,
                    "remaining_steps": remaining_steps,
                    "next_steps": remaining_steps[:3],
                    "last_execution_result": result,
                    "debug_attempts": debug_attempts + 1,
                },
            )
            return {
                "ready_to_test": False,
                "completed_steps": completed_steps,
                "resume_debug_loop": True,
            }

        self.store.upsert_artifact(
            thread_id=thread_id,
            kind="project_state",
            title="Project State",
            summary="Approved execution failed.",
            content={
                **project_state,
                "phase": "blocked",
                "awaiting_user_feedback": True,
                "latest_status": "failed",
                "ready_to_test": False,
                "last_execution_result": result,
                "debug_attempts": debug_attempts + (1 if command_kind == "python_execution" and pending_step is not None else 0),
            },
        )
        return {"ready_to_test": False, "completed_steps": completed_steps}

    def _continue_autonomous_run_after_approval(self, thread_id: str) -> dict | None:
        project_state = self._get_project_state(thread_id)
        if not project_state or not project_state.get("autonomous_mode"):
            return None
        if project_state.get("ready_to_test") or project_state.get("pending_step"):
            return None
        if not (project_state.get("remaining_steps") or []):
            return None

        history = self.store.list_messages(thread_id)
        user_message = project_state.get("last_user_request") or "Continue the active project."
        user_message_for_workers = self.language_policy.normalize_for_internal_agents(
            user_message,
            self.language_policy.detect_user_language(user_message),
        )
        route = RouteDecision(
            decision=ManagerDecision.CODING,
            reason="Continue the autonomous project after an approved execution.",
        )
        execution = self._execute_worker_route(
            thread_id=thread_id,
            user_message=user_message_for_workers,
            history=history,
            route=route,
            planning_note=None,
            fallback_internal_message=(
                "Continue the active project after the approved execution. "
                "Use the stored repo plan, remaining steps, and latest execution result."
            ),
            user_language="en",
        )
        reply = execution["reply"]
        message = self.store.add_message(
            thread_id,
            role="assistant",
            content=reply,
            agent="manager",
            meta={
                "route": "coding",
                "approval_request_id": None,
                "manager_source": "approval_resume",
                "final_decision": "coding",
                "llm_decision": "coding",
                "language_policy": {
                    "user_language": "en",
                    "internal_language": "en",
                },
                "internal_payload": execution["delegated_result"].get("internal_payload"),
            },
        )
        return {
            "message_id": message["id"],
            "reply": reply,
            "approval_request": execution["delegated_result"].get("approval_request"),
        }

    def approve_execution(self, approval_id: str) -> dict:
        approval = self.store.get_approval_request(approval_id)
        if approval is None:
            raise KeyError("Approval Request nicht gefunden.")
        if approval["status"] != ApprovalStatus.PENDING.value:
            return approval
        log_event(
            logger,
            "approval_approved",
            thread_id=approval["thread_id"],
            approval_id=approval_id,
            command_preview=approval["command"].get("preview"),
        )

        self.store.update_approval(
            approval_id,
            ApprovalStatus.APPROVED,
            decision_note="Vom Nutzer freigegeben.",
        )
        try:
            result = execute_python_approval(approval["command"])
            final_status = ApprovalStatus.EXECUTED if result["returncode"] == 0 else ApprovalStatus.FAILED
            approval = self.store.update_approval(
                approval_id,
                final_status,
                decision_note="Freigabe erteilt und Ausführung abgeschlossen.",
                result=result,
            )
        except (WorkspaceSecurityError, ExecutionPolicyError) as exc:
            result = {
                "command": approval["command"],
                "returncode": -1,
                "stdout": "",
                "stderr": getattr(exc, "user_message", str(exc)),
                "error_code": getattr(exc, "code", "execution_policy_denied"),
            }
            approval = self.store.update_approval(
                approval_id,
                ApprovalStatus.FAILED,
                decision_note="Freigabe erteilt, aber Sicherheitsregeln haben die Ausführung blockiert.",
                result=result,
            )
        except Exception as exc:
            result = {
                "command": approval["command"],
                "returncode": -1,
                "stdout": "",
                "stderr": str(exc),
            }
            approval = self.store.update_approval(
                approval_id,
                ApprovalStatus.FAILED,
                decision_note="Freigabe erteilt, aber Ausführung ist fehlgeschlagen.",
                result=result,
            )
        log_event(
            logger,
            "approval_execution_finished",
            thread_id=approval["thread_id"],
            approval_id=approval_id,
            status=approval["status"],
            backend=result.get("sandbox_backend"),
            returncode=result.get("returncode"),
        )
        command_kind = result.get("command_kind") or approval["command"].get("kind")
        if approval["status"] == ApprovalStatus.EXECUTED.value:
            if command_kind == "pip_install":
                packages = ", ".join(result.get("packages") or approval["command"].get("packages") or [])
                note = (
                    f"Dependency-Installation ausgeführt: `{approval['command']['preview']}`\n"
                    f"Pakete: {packages}\n"
                    f"Returncode: {result['returncode']}"
                )
            else:
                note = (
                    f"Ausgeführt: `{approval['command']['preview']}`\n"
                    f"Returncode: {result['returncode']}"
                )
        else:
            prefix = "Dependency-Installation fehlgeschlagen oder blockiert" if command_kind == "pip_install" else "Ausführung fehlgeschlagen oder blockiert"
            note = (
                f"{prefix}: `{approval['command']['preview']}`\n"
                f"Grund: {result.get('stderr', 'unbekannt')}"
            )
        self.store.add_message(
            approval["thread_id"],
            role="assistant",
            content=note,
            agent="manager",
            meta={"approval_request_id": approval_id, "execution_result": result},
        )
        follow_up = self._mark_autonomous_approval_outcome(approval["thread_id"], result)
        if follow_up is None:
            self._update_project_state_after_execution_approval(approval["thread_id"], result)
        if approval["status"] == ApprovalStatus.EXECUTED.value:
            continuation = self._continue_autonomous_run_after_approval(approval["thread_id"])
            if continuation is not None:
                approval = dict(approval)
                approval["continuation"] = continuation
        elif follow_up is not None and follow_up.get("resume_debug_loop"):
            continuation = self._continue_autonomous_run_after_approval(approval["thread_id"])
            approval = dict(approval)
            approval["continuation"] = continuation
        elif follow_up is not None:
            approval = dict(approval)
            approval["continuation"] = None
        return approval

    def reject_execution(self, approval_id: str) -> dict:
        approval = self.store.get_approval_request(approval_id)
        if approval is None:
            raise KeyError("Approval Request nicht gefunden.")
        if approval["status"] != ApprovalStatus.PENDING.value:
            return approval
        approval = self.store.update_approval(
            approval_id,
            ApprovalStatus.REJECTED,
            decision_note="Vom Nutzer abgelehnt.",
        )
        log_event(
            logger,
            "approval_rejected",
            thread_id=approval["thread_id"],
            approval_id=approval_id,
            command_preview=approval["command"].get("preview"),
        )
        self.store.add_message(
            approval["thread_id"],
            role="assistant",
            content=f"Ausführung abgelehnt: `{approval['command']['preview']}`",
            agent="manager",
            meta={"approval_request_id": approval_id},
        )
        return approval

    def _direct_reply(
        self,
        user_message: str,
        user_language: str,
        planning_note: dict | None,
    ) -> tuple[str, str, str | None]:
        plan = self._llm_plan(planning_note)
        if plan is not None and self._is_plausible_manager_reply(plan.user_reply):
            return plan.user_reply, "ollama", None
        raise ValueError("Manager returned an implausible direct reply.")

    def _execute_project_planning(
        self,
        thread_id: str,
        user_message: str,
        history: list[dict],
        planning_note: dict | None,
        user_language: str,
        pending_message_id: int | None = None,
    ) -> dict:
        if pending_message_id is not None:
            self._update_pending_status(
                pending_message_id=pending_message_id,
                comment=self.language_policy.user_text(
                    user_language,
                    "Research-Agent arbeitet am Projektplan",
                    "Research agent is shaping the project plan",
                ),
            )

        base_task = self._worker_task(thread_id, user_message, planning_note, user_message)
        research_task = (
            "Prepare a project plan before implementation starts. "
            "Focus on scope, feasibility, architecture direction, suggested repo structure, "
            "implementation steps, dependencies, risks, and the best first increment. "
            "End with the most useful questions for the manager to bring back to the user.\n\n"
            f"Task: {base_task}"
        )
        research_result = self.delegation.execute(
            ManagerDecision.RESEARCH,
            thread_id,
            user_message,
            history,
            internal_task=research_task,
        )

        if pending_message_id is not None:
            self._update_pending_status(
                pending_message_id=pending_message_id,
                comment=self.language_policy.user_text(
                    user_language,
                    "Kritiker-Agent prüft den Projektplan",
                    "Reviewer agent is stress-testing the project plan",
                ),
            )

        research_payload = research_result.get("internal_payload") or {}
        review_task = (
            "Critique this draft project plan before any coding starts. "
            "Challenge oversized scope, missing phases, weak validation strategy, "
            "and places where the manager should ask the user for feedback before implementation.\n\n"
            f"Original task: {base_task}\n"
            f"Research summary: {research_result.get('internal_summary', '')}\n"
            f"Research findings: {research_payload.get('findings', [])}\n"
            f"Research open questions: {research_payload.get('open_questions', [])}\n"
            f"Research recommendation: {research_payload.get('recommendation', '')}"
        )
        review_result = self.delegation.execute(
            ManagerDecision.REVIEW,
            thread_id,
            user_message,
            history,
            internal_task=review_task,
        )

        project_plan = self._build_project_plan_payload(
            user_message=user_message,
            planning_note=planning_note,
            research_result=research_result,
            review_result=review_result,
        )
        self._store_project_plan_artifacts(thread_id, project_plan)
        reply = self._render_project_plan_reply(user_language, planning_note, project_plan)
        return {
            "status": "completed",
            "reply": reply,
            "user_reply": reply,
            "internal_summary": project_plan["summary"],
            "tool_results": [],
            "internal_payload": {
                "language": "en",
                "thread_id": thread_id,
                "task": f"plan: {base_task}",
                "policy": "Project planning is manager-led. Use stored project memory and wait for user feedback before broad implementation.",
                "source": "manager_plan",
                "summary": project_plan["summary"],
                "project_plan": project_plan,
            },
        }

    def _build_project_plan_payload(
        self,
        user_message: str,
        planning_note: dict | None,
        research_result: dict,
        review_result: dict,
    ) -> dict:
        plan = self._llm_plan(planning_note)
        outline = self._llm_project_outline(planning_note)
        research_payload = research_result.get("internal_payload") or {}
        review_status = review_result.get("status")
        review_payload = (review_result.get("internal_payload") or {}) if review_status == "completed" else {}
        review_summary = review_result.get("internal_summary", "") if review_status == "completed" else ""

        approach = self._dedupe_items(
            research_payload.get("findings") or [research_result.get("internal_summary", "")]
        )[:4]
        risks = self._dedupe_items(review_payload.get("findings") or [review_summary])[:4]
        open_questions = self._dedupe_items(
            (research_payload.get("open_questions") or []) + (review_payload.get("open_questions") or [])
        )[:4]
        next_steps = self._dedupe_items(
            [research_payload.get("recommendation", ""), review_payload.get("recommendation", "")]
        )[:3]
        outlined_steps = self._dedupe_items((outline.steps if outline is not None else []) + next_steps)
        summary = (
            review_summary
            or research_result.get("internal_summary")
            or (plan.summary if plan is not None else "")
            or "Project plan prepared."
        )
        return {
            "user_message": user_message,
            "manager_summary": plan.summary if plan is not None else summary,
            "summary": summary,
            "reason": plan.reason if plan is not None else "Project planning requested.",
            "manager_reply_hint": plan.user_reply if plan is not None else "",
            "approach": approach,
            "repo_structure": list(outline.repo_structure) if outline is not None else [],
            "risks": risks,
            "open_questions": open_questions,
            "steps": outlined_steps,
            "next_steps": outlined_steps[:3],
            "validation_steps": list(outline.validation_steps) if outline is not None else [],
            "completion_criteria": list(outline.completion_criteria) if outline is not None else [],
            "project_outline": self._project_outline_dict(outline),
            "research": {
                "summary": research_result.get("internal_summary", ""),
                "reply": research_result.get("reply", ""),
                "findings": research_payload.get("findings", []),
                "assumptions": research_payload.get("assumptions", []),
                "open_questions": research_payload.get("open_questions", []),
                "recommendation": research_payload.get("recommendation", ""),
                "sources": research_payload.get("sources", []),
            },
            "review": {
                "summary": review_result.get("internal_summary", ""),
                "reply": review_result.get("reply", ""),
                "status": review_status,
                "findings": review_payload.get("findings", []),
                "assumptions": review_payload.get("assumptions", []),
                "open_questions": review_payload.get("open_questions", []),
                "recommendation": review_payload.get("recommendation", ""),
            },
            "awaiting_user_feedback": True,
        }

    def _store_project_plan_artifacts(self, thread_id: str, project_plan: dict) -> None:
        self.store.upsert_artifact(
            thread_id=thread_id,
            kind="project_plan",
            title="Project Plan",
            summary=project_plan.get("summary", "") or "Project plan prepared.",
            content=project_plan,
        )
        self.store.upsert_artifact(
            thread_id=thread_id,
            kind="project_brief",
            title="Project Brief",
            summary=project_plan.get("manager_summary", "") or project_plan.get("summary", ""),
            content={
                "last_user_request": project_plan.get("user_message", ""),
                "summary": project_plan.get("summary", ""),
                "approach": project_plan.get("approach", []),
                "repo_structure": project_plan.get("repo_structure", []),
                "risks": project_plan.get("risks", []),
                "steps": project_plan.get("steps", []),
                "next_steps": project_plan.get("next_steps", []),
                "validation_steps": project_plan.get("validation_steps", []),
                "completion_criteria": project_plan.get("completion_criteria", []),
            },
        )
        self.store.upsert_artifact(
            thread_id=thread_id,
            kind="project_state",
            title="Project State",
            summary="Planning complete. Waiting for user feedback or approval to start the next step.",
            content={
                "phase": "planning",
                "awaiting_user_feedback": True,
                "last_summary": project_plan.get("summary", ""),
                "repo_structure": project_plan.get("repo_structure", []),
                "steps": project_plan.get("steps", []),
                "next_steps": project_plan.get("next_steps", []),
                "remaining_steps": project_plan.get("steps", []),
                "completed_steps": [],
                "open_questions": project_plan.get("open_questions", []),
                "validation_steps": project_plan.get("validation_steps", []),
                "completion_criteria": project_plan.get("completion_criteria", []),
                "ready_to_test": False,
                "autonomous_mode": False,
            },
        )
        if project_plan.get("research"):
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="research_notes",
                title="Research Notes",
                summary=project_plan["research"].get("summary", "") or "Latest planning research notes.",
                content=project_plan["research"],
            )
            if project_plan["research"].get("sources"):
                self.store.upsert_artifact(
                    thread_id=thread_id,
                    kind="web_research_notes",
                    title="Web Research Notes",
                    summary=project_plan["research"].get("summary", "") or "Latest web-backed planning research.",
                    content=project_plan["research"],
                )
        if project_plan.get("review"):
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="review_notes",
                title="Review Notes",
                summary=project_plan["review"].get("summary", "") or "Latest plan review.",
                content=project_plan["review"],
            )

    def _render_project_plan_reply(
        self,
        user_language: str,
        planning_note: dict | None,
        project_plan: dict,
    ) -> str:
        plan = self._llm_plan(planning_note)
        intro = (plan.user_reply if plan is not None else "").strip()
        if not intro:
            intro = self.language_policy.user_text(
                user_language,
                "Ich habe das Projekt zuerst intern strukturiert und gegenprüfen lassen, bevor wir breit ins Coding gehen.",
                "I first structured the project internally and had that plan challenged before we go broad on coding.",
            )

        lines = [intro]
        if project_plan.get("approach"):
            title = "Mein aktueller Vorschlag" if user_language == "de" else "Current proposal"
            lines.append(self._format_section(title, project_plan["approach"], numbered=False))
        if project_plan.get("repo_structure"):
            title = "Vorgeschlagene Repo-Struktur" if user_language == "de" else "Suggested repo structure"
            lines.append(self._format_section(title, project_plan["repo_structure"], numbered=False))
        if project_plan.get("risks"):
            title = "Worauf wir achten sollten" if user_language == "de" else "Things to watch"
            lines.append(self._format_section(title, project_plan["risks"], numbered=False))
        if project_plan.get("steps"):
            title = "Geplante Schritte" if user_language == "de" else "Planned steps"
            lines.append(self._format_section(title, project_plan["steps"], numbered=True))
        if project_plan.get("validation_steps"):
            title = "Validierung" if user_language == "de" else "Validation"
            lines.append(self._format_section(title, project_plan["validation_steps"], numbered=False))
        if project_plan.get("open_questions"):
            title = "Punkte für dein Feedback" if user_language == "de" else "Points for your feedback"
            lines.append(self._format_section(title, project_plan["open_questions"], numbered=False))

        close = self.language_policy.user_text(
            user_language,
            "Wenn dir die Richtung gefällt, setze ich im nächsten Schritt den ersten kleinen Step um und halte dich danach wieder mit Vorschlag und Fortschritt auf dem Laufenden.",
            "If this direction looks good to you, I will implement the first small step next and then come back with progress and the next recommendation.",
        )
        lines.append(close)
        return "\n\n".join(line for line in lines if line.strip())

    def _manager_wrap(
        self,
        reason: str,
        delegated_result: dict,
        planning_note: dict | None,
        user_language: str,
        user_message: str,
        consultation: dict | None = None,
        implementation_review: dict | None = None,
    ) -> str:
        status = delegated_result.get("status")
        reply = self._sanitize_user_reply(delegated_result, user_message, user_language)
        if status in {"blocked", "error"}:
            return reply
        internal_payload = delegated_result.get("internal_payload") or {}
        manager_prefix = self._delegation_prefix(user_language, delegated_result)
        consultation_note = ""
        if consultation:
            consultation_note = self.language_policy.user_text(
                user_language,
                "Ich habe die Aufgabe zuerst intern in kleinere Schritte und Risiken zerlegt.",
                "I first broke the task down internally into smaller steps and risks.",
            )
        status_note = self._user_facing_step_status_note(
            user_language=user_language,
            delegated_result=delegated_result,
            implementation_review=implementation_review,
        )
        review_note = ""
        if implementation_review and implementation_review.get("summary"):
            review_note = self.language_policy.user_text(
                user_language,
                "Ich habe den letzten Coding-Schritt zusätzlich intern gegenprüfen lassen.",
                "I also had the latest coding step checked internally.",
            )
        repair_note = ""
        repair_loop = internal_payload.get("repair_loop") or {}
        if int(repair_loop.get("attempted") or 0) > 0:
            repair_note = self.language_policy.user_text(
                user_language,
                f"Der Manager hat daraufhin noch {int(repair_loop.get('attempted') or 0)} fokussierte Reparatur-Schritte nachgeschoben.",
                f"The manager then ran {int(repair_loop.get('attempted') or 0)} focused repair steps.",
            )
        autonomous_note = ""
        autonomous_payload = internal_payload.get("autonomous_run") or {}
        if autonomous_payload.get("enabled"):
            completed = autonomous_payload.get("steps_completed") or []
            total = autonomous_payload.get("steps_total") or []
            if autonomous_payload.get("ready_to_test"):
                autonomous_note = self.language_policy.user_text(
                    user_language,
                    f"Der Manager hat die geplanten Schritte autonom bis zu einem ersten testbaren Stand abgearbeitet ({len(completed)}/{len(total)} Schritte abgeschlossen).",
                    f"The manager worked through the planned steps autonomously until the project reached an initial ready-to-test state ({len(completed)}/{len(total)} steps completed).",
                )
            else:
                autonomous_note = self.language_policy.user_text(
                    user_language,
                    f"Der Manager hat in diesem Lauf mehrere geplante Schritte autonom ausgeführt ({len(completed)}/{len(total)} Schritte abgeschlossen).",
                    f"The manager executed multiple planned steps autonomously in this run ({len(completed)}/{len(total)} steps completed).",
                )
        if status == "completed":
            return "\n".join(part for part in (consultation_note, status_note, review_note, repair_note, autonomous_note, manager_prefix, reply) if part).strip()
        if status == "approval_required":
            return "\n".join(part for part in (consultation_note, status_note, review_note, repair_note, autonomous_note, manager_prefix, reply) if part).strip()
        if status == "completed_with_approval":
            return "\n".join(part for part in (consultation_note, status_note, review_note, repair_note, autonomous_note, manager_prefix, reply) if part).strip()
        return self.language_policy.user_text(
            user_language,
            f"Delegiert: {reason}\n{reply}",
            f"Delegated: {reason}\n{reply}",
        )

    def _user_facing_step_status_note(
        self,
        *,
        user_language: str,
        delegated_result: dict,
        implementation_review: dict | None,
    ) -> str:
        review_payload = (implementation_review or {}).get("internal_payload") or {}
        step_outcome = str((delegated_result.get("internal_payload") or {}).get("step_outcome", "")).strip() or self._classify_step_outcome(delegated_result, implementation_review)
        review_verdict = str(review_payload.get("verdict", "")).strip().lower()
        project_status = str(review_payload.get("project_status", "")).strip().lower()
        if review_verdict == "blocked" or delegated_result.get("status") in {"blocked", "error"}:
            return self.language_policy.user_text(
                user_language,
                "Status: Der Schritt ist aktuell blockiert.",
                "Status: This step is currently blocked.",
            )
        if step_outcome == "analysis_only":
            return self.language_policy.user_text(
                user_language,
                "Status: Ich habe den aktuellen Stand analysiert, aber in diesem Schritt noch keine Codeänderung umgesetzt.",
                "Status: I analyzed the current state, but this step has not applied a code change yet.",
            )
        if step_outcome == "no_effect":
            return self.language_policy.user_text(
                user_language,
                "Status: Es wurde zwar ein Workspace-Schritt ausgeführt, aber die angeforderte Implementierung ist im Ergebnis noch nicht vorhanden.",
                "Status: A workspace step ran, but the requested implementation is still not present in the result.",
            )
        if review_verdict == "needs_repair":
            return self.language_policy.user_text(
                user_language,
                "Status: Code wurde geändert, aber der Schritt braucht noch eine fokussierte Nachbesserung.",
                "Status: Code changed, but the step still needs a focused repair.",
            )
        if project_status in {"validated_ready_to_test", "ready_to_test", "project_done"}:
            return self.language_policy.user_text(
                user_language,
                "Status: Der Schritt ist umgesetzt und mit der aktuellen Evidenz testbereit.",
                "Status: The step is implemented and the current evidence is ready to test.",
            )
        if step_outcome == "validation_requested":
            return self.language_policy.user_text(
                user_language,
                "Status: Für diesen Schritt wurde Validierung vorbereitet, aber es gab in diesem Lauf keine neue Codeänderung.",
                "Status: Validation was prepared for this step, but this pass did not introduce a new code change.",
            )
        if step_outcome == "implemented":
            return self.language_policy.user_text(
                user_language,
                "Status: Der Schritt wurde umgesetzt, ist aber noch nicht als testbereit validiert.",
                "Status: The step was implemented, but it is not yet validated as ready to test.",
            )
        return ""

    def _user_explicitly_requested_raw_contents(self, user_message: str) -> bool:
        lowered = user_message.lower()
        return any(re.search(pattern, lowered) for pattern in RAW_CONTENT_REQUEST_PATTERNS)

    def _action_message_for_user(
        self,
        action: dict,
        *,
        user_language: str,
        allow_raw_contents: bool,
    ) -> str:
        action_type = str(action.get("action_type", "")).strip()
        target = str(action.get("target") or "").strip()
        if action_type == "read_file" and target and not allow_raw_contents:
            return self.language_policy.user_text(
                user_language,
                f"Datei geprüft: `{target}`.",
                f"Inspected file: `{target}`.",
            )
        message = str(action.get("message", "")).strip()
        if action_type == "read_file" and target and allow_raw_contents and message:
            return message
        if message:
            return message
        if action_type == "create_file" and target:
            return self.language_policy.user_text(
                user_language,
                f"Datei erstellt: `{target}`.",
                f"File created: `{target}`.",
            )
        if action_type == "make_directory" and target:
            return self.language_policy.user_text(
                user_language,
                f"Ordner erstellt: `{target}`.",
                f"Directory created: `{target}`.",
            )
        if action_type == "delete_path" and target:
            return self.language_policy.user_text(
                user_language,
                f"Pfad gelöscht: `{target}`.",
                f"Deleted path: `{target}`.",
            )
        if action_type == "request_execution" and target:
            return self.language_policy.user_text(
                user_language,
                f"Ausführungsantrag vorbereitet: `{target}`.",
                f"Execution request prepared: `{target}`.",
            )
        return ""

    def _sanitize_user_reply(self, delegated_result: dict, user_message: str, user_language: str) -> str:
        allow_raw_contents = self._user_explicitly_requested_raw_contents(user_message)
        if allow_raw_contents:
            return delegated_result.get("reply", "")

        actions_executed = delegated_result.get("actions_executed") or []
        actions_blocked = delegated_result.get("actions_blocked") or []
        user_messages: list[str] = []
        for action in actions_executed:
            if not isinstance(action, dict):
                continue
            message = self._action_message_for_user(
                action,
                user_language=user_language,
                allow_raw_contents=allow_raw_contents,
            )
            if message:
                user_messages.append(message)
        for action in actions_blocked:
            if not isinstance(action, dict):
                continue
            message = self._action_message_for_user(
                action,
                user_language=user_language,
                allow_raw_contents=allow_raw_contents,
            )
            if message:
                user_messages.append(message)
        approval_request = delegated_result.get("approval_request")
        if isinstance(approval_request, dict):
            approval_message = str(approval_request.get("user_message", "")).strip()
            if approval_message:
                user_messages.append(approval_message)
        user_constraint_note = str(delegated_result.get("user_constraint_note", "")).strip()
        if user_constraint_note:
            user_messages.append(user_constraint_note)
        if not user_messages:
            return delegated_result.get("reply", "")
        return "\n".join(self._dedupe_items(user_messages))

    def _approval_block_reason(
        self,
        approval_request: dict | None,
        constraints: dict,
        user_language: str,
    ) -> str:
        if not approval_request or not constraints:
            return ""
        tool_name = str(approval_request.get("tool_name", "")).strip()
        preview = str((approval_request.get("command") or {}).get("preview", "")).strip().lower()
        if tool_name == "install_python_requirements" and constraints.get("forbid_dependency_install"):
            return self.language_policy.user_text(
                user_language,
                "Keine Paketinstallation vorbereitet, weil du Dependency-Installationen derzeit ausdrücklich ausgeschlossen hast.",
                "No package installation was prepared because you explicitly ruled out dependency installs for now.",
            )
        if tool_name == "request_python_execution":
            if constraints.get("forbid_pytest") and "pytest" in preview:
                return self.language_policy.user_text(
                    user_language,
                    "Kein Pytest-Lauf vorbereitet, weil du Pytest derzeit ausdrücklich ausgeschlossen hast.",
                    "No pytest run was prepared because you explicitly ruled out pytest for now.",
                )
            if constraints.get("forbid_validation_execution"):
                return self.language_policy.user_text(
                    user_language,
                    "Keine Ausführung vorbereitet, weil du Validierungs- oder Laufzeit-Ausführungen derzeit ausdrücklich ausgeschlossen hast.",
                    "No execution was prepared because you explicitly ruled out validation/runtime executions for now.",
                )
        return ""

    def _enforce_manager_constraints_on_result(
        self,
        *,
        thread_id: str,
        user_message: str,
        delegated_result: dict,
        user_language: str,
    ) -> dict:
        constraints = self._get_manager_constraints(thread_id)
        if not constraints:
            return delegated_result

        approval_request = delegated_result.get("approval_request")
        block_reason = self._approval_block_reason(approval_request, constraints, user_language)
        if not block_reason:
            return delegated_result

        internal_payload = dict(delegated_result.get("internal_payload") or {})
        internal_payload["manager_constraints"] = constraints
        internal_payload["suppressed_approval_request"] = {
            "tool_name": approval_request.get("tool_name"),
            "preview": (approval_request.get("command") or {}).get("preview"),
            "reason": block_reason,
        }
        self_check = dict(internal_payload.get("self_check") or {})
        follow_up = list(self_check.get("follow_up") or [])
        if block_reason not in follow_up:
            follow_up.append(block_reason)
        self_check["follow_up"] = follow_up
        internal_payload["self_check"] = self_check

        actions_executed = delegated_result.get("actions_executed") or []
        if approval_request.get("tool_name") == "request_python_execution":
            delegated_result["actions_executed"] = [
                action
                for action in actions_executed
                if str((action or {}).get("action_type", "")).strip() != "request_execution"
            ]

        delegated_result["approval_request"] = None
        delegated_result["approval_request_created"] = False
        delegated_result["status"] = "completed"
        delegated_result["user_constraint_note"] = block_reason
        delegated_result["internal_payload"] = internal_payload
        delegated_result["reply"] = self._sanitize_user_reply(delegated_result, user_message, user_language)
        delegated_result["user_reply"] = delegated_result["reply"]
        return delegated_result

    def _execute_coding_step(
        self,
        *,
        thread_id: str,
        user_message: str,
        history: list[dict],
        worker_task: str,
        coding_structured_plan: CodingDelegationPlan | None,
        user_language: str,
        consultation: dict | None,
        pending_message_id: int | None = None,
    ) -> tuple[dict, dict | None]:
        explorer_context = self._prepare_explorer_context(
            thread_id=thread_id,
            user_message=user_message,
            history=history,
            worker_task=worker_task,
            user_language=user_language,
            pending_message_id=pending_message_id,
        )
        step_contract = self._build_coding_step_contract(
            thread_id=thread_id,
            user_message=user_message,
            worker_task=worker_task,
            coding_structured_plan=coding_structured_plan,
            consultation=consultation,
            explorer_context=explorer_context,
        )
        worker_task = self._augment_coding_task_with_contract(worker_task, step_contract)
        delegated_result = self.delegation.execute(
            ManagerDecision.CODING,
            thread_id,
            user_message,
            history,
            internal_task=worker_task,
            structured_plan=coding_structured_plan,
        )
        if consultation is not None:
            internal_payload = dict(delegated_result.get("internal_payload") or {})
            internal_payload["consultation"] = consultation
            delegated_result["internal_payload"] = internal_payload
        internal_payload = dict(delegated_result.get("internal_payload") or {})
        if explorer_context is not None:
            internal_payload["explorer_context"] = explorer_context
        internal_payload["step_contract"] = step_contract
        delegated_result["internal_payload"] = internal_payload
        implementation_review = self._review_coding_result(
            thread_id=thread_id,
            user_message=user_message,
            history=history,
            delegated_result=delegated_result,
            user_language=user_language,
            pending_message_id=pending_message_id,
        )
        implementation_review = self._apply_review_evidence_gate(delegated_result, implementation_review)
        internal_payload = dict(delegated_result.get("internal_payload") or {})
        internal_payload["step_outcome"] = self._classify_step_outcome(delegated_result, implementation_review)
        delegated_result["internal_payload"] = internal_payload
        if implementation_review is not None:
            internal_payload = dict(delegated_result.get("internal_payload") or {})
            internal_payload["implementation_review"] = implementation_review
            delegated_result["internal_payload"] = internal_payload
        delegated_result = self._apply_dependency_gate(
            thread_id=thread_id,
            delegated_result=delegated_result,
            user_language=user_language,
        )
        self._store_delegation_artifacts(
            thread_id,
            ManagerDecision.CODING,
            user_message,
            delegated_result,
            consultation,
            implementation_review=implementation_review,
        )
        return delegated_result, implementation_review

    def _should_attempt_repair_loop(
        self,
        *,
        thread_id: str,
        user_message: str,
        delegated_result: dict,
        implementation_review: dict | None,
    ) -> bool:
        if implementation_review is None:
            return False
        if delegated_result.get("status") != "completed":
            return False
        if delegated_result.get("approval_request") is not None:
            return False
        review_payload = implementation_review.get("internal_payload") or {}
        verdict = str(review_payload.get("verdict", "")).strip().lower()
        definition_of_done_met = review_payload.get("definition_of_done_met")
        repair_tasks = review_payload.get("repair_tasks") or []
        constraints = self._get_manager_constraints(thread_id)
        if constraints.get("stop_after_step"):
            return False
        validation_like_tasks = [task for task in repair_tasks if self._repair_task_looks_like_validation(task)]
        if validation_like_tasks and constraints and (
            constraints.get("forbid_dependency_install")
            or constraints.get("forbid_pytest")
            or constraints.get("forbid_validation_execution")
        ):
            implementation_tasks = [task for task in repair_tasks if not self._repair_task_looks_like_validation(task)]
            if not implementation_tasks:
                return False
        return bool(repair_tasks) and (verdict == "needs_repair" or definition_of_done_met is False)

    def _repair_task_looks_like_validation(self, task: str) -> bool:
        lowered = str(task or "").strip().lower()
        if not lowered:
            return False
        validation_tokens = (
            "pytest",
            "test",
            "validation",
            "smoke",
            "run ",
            "execute",
            "ausführ",
            "install",
            "dependency",
            "package",
            "pip",
        )
        return any(token in lowered for token in validation_tokens)

    def _build_repair_worker_task(
        self,
        *,
        original_worker_task: str,
        user_message: str,
        delegated_result: dict,
        implementation_review: dict,
        attempt_index: int,
    ) -> str:
        internal_payload = delegated_result.get("internal_payload") or {}
        review_payload = implementation_review.get("internal_payload") or {}
        step_contract = internal_payload.get("step_contract") or {}
        repair_tasks = review_payload.get("repair_tasks") or []
        findings = review_payload.get("findings") or []
        touched_paths = self._dedupe_items(((internal_payload.get("self_check") or {}).get("touched_paths") or []))
        changed_paths = self._dedupe_items(
            [
                str(item.get("path", "")).strip()
                for item in (review_payload.get("snapshot_changes") or [])
                if str(item.get("path", "")).strip() and str(item.get("status", "")).strip().lower() != "unchanged"
            ]
        )
        narrowed_paths = self._dedupe_items(
            touched_paths + changed_paths + list(step_contract.get("relevant_paths") or []) + list(step_contract.get("validation_relevant_paths") or [])
        )[:6]
        failing_dod = self._dedupe_items(
            list(step_contract.get("failing_definition_of_done") or []) + repair_tasks
        )[:4]
        return (
            "Implement a focused repair step for the latest coding result. "
            "Do not restart the feature and do not reopen unrelated roadmap work. "
            "Only fix the remaining issues below while preserving the current structure.\n\n"
            f"Original user request: {user_message}\n"
            f"Manager step goal: {step_contract.get('goal', '')}\n"
            f"Latest request kind: {step_contract.get('request_kind', 'repair_request')}\n"
            f"Focused change summary: {step_contract.get('change_request_summary', '')}\n"
            f"Narrow repair paths: {narrowed_paths}\n"
            f"Failed definition-of-done items: {failing_dod}\n"
            f"Latest coding summary: {delegated_result.get('internal_summary', '')}\n"
            f"Reviewer verdict: {review_payload.get('verdict', '')}\n"
            f"Reviewer findings: {findings}\n"
            f"Concrete repair tasks: {repair_tasks}\n"
            f"Repair attempt: {attempt_index}/{MANAGER_AUTONOMOUS_MAX_DEBUG_REPAIRS}\n\n"
            "After the fixes, leave the step coherent, keep the scope smaller than the previous main step, and make the repair visible in the after-state."
        ).strip()

    def _run_manager_repair_loop(
        self,
        *,
        thread_id: str,
        user_message: str,
        history: list[dict],
        worker_task: str,
        user_language: str,
        consultation: dict | None,
        delegated_result: dict,
        implementation_review: dict | None,
        pending_message_id: int | None = None,
    ) -> tuple[dict, dict | None]:
        if not self._should_attempt_repair_loop(
            thread_id=thread_id,
            user_message=user_message,
            delegated_result=delegated_result,
            implementation_review=implementation_review,
        ):
            return delegated_result, implementation_review

        repair_history: list[dict] = []
        current_result = delegated_result
        current_review = implementation_review

        for attempt in range(1, 2):
            if not self._should_attempt_repair_loop(
                thread_id=thread_id,
                user_message=user_message,
                delegated_result=current_result,
                implementation_review=current_review,
            ):
                break

            if pending_message_id is not None:
                self._update_pending_status(
                    pending_message_id=pending_message_id,
                    comment=self.language_policy.user_text(
                        user_language,
                        "Manager leitet einen fokussierten Reparatur-Schritt ein",
                        "Manager is triggering a focused repair step",
                    ),
                )

            repair_task = self._build_repair_worker_task(
                original_worker_task=worker_task,
                user_message=user_message,
                delegated_result=current_result,
                implementation_review=current_review,
                attempt_index=attempt,
            )
            repair_result, repair_review = self._execute_coding_step(
                thread_id=thread_id,
                user_message=user_message,
                history=history,
                worker_task=repair_task,
                coding_structured_plan=None,
                user_language=user_language,
                consultation=consultation,
                pending_message_id=pending_message_id,
            )
            repair_payload = (repair_review or {}).get("internal_payload") or {}
            repair_history.append(
                {
                    "attempt": attempt,
                    "status": repair_result.get("status"),
                    "review_verdict": repair_payload.get("verdict"),
                    "definition_of_done_met": repair_payload.get("definition_of_done_met"),
                    "repair_tasks": repair_payload.get("repair_tasks", []),
                    "summary": repair_result.get("internal_summary", ""),
                }
            )
            current_result = repair_result
            current_review = repair_review
            if current_result.get("status") in {"blocked", "error", "approval_required", "completed_with_approval"}:
                break

        repair_loop_payload = {
            "attempted": len(repair_history),
            "runs": repair_history,
            "stopped_with_review": (current_review or {}).get("internal_payload", {}).get("verdict"),
        }
        internal_payload = dict(current_result.get("internal_payload") or {})
        internal_payload["repair_loop"] = repair_loop_payload
        current_result["internal_payload"] = internal_payload
        return current_result, current_review

    def _apply_dependency_gate(
        self,
        *,
        thread_id: str,
        delegated_result: dict,
        user_language: str,
    ) -> dict:
        status = delegated_result.get("status")
        if status not in {"completed", "completed_with_approval", "approval_required"}:
            return delegated_result

        dependency_check = analyze_workspace_dependencies(thread_id)
        if not dependency_check.get("python_files"):
            return delegated_result

        missing_requirements = self._dedupe_items(dependency_check.get("missing_requirements") or [])
        missing_installations = self._dedupe_items(dependency_check.get("missing_installations") or [])
        packages_to_install = self._dedupe_items(dependency_check.get("packages_to_install") or [])

        dependency_check["summary"] = self._dependency_check_summary(
            user_language=user_language,
            missing_requirements=missing_requirements,
            missing_installations=missing_installations,
            packages_to_install=packages_to_install,
        )
        dependency_check["follow_up"] = self._dependency_follow_up(
            user_language=user_language,
            packages_to_install=packages_to_install,
        )

        internal_payload = dict(delegated_result.get("internal_payload") or {})
        internal_payload["dependency_check"] = dependency_check

        self_check = dict(internal_payload.get("self_check") or {})
        existing_follow_up = list(self_check.get("follow_up") or [])
        for item in dependency_check["follow_up"]:
            if item not in existing_follow_up:
                existing_follow_up.append(item)
        self_check["follow_up"] = existing_follow_up
        internal_payload["self_check"] = self_check
        delegated_result["internal_payload"] = internal_payload

        if not packages_to_install:
            return delegated_result

        rationale = (
            "Install the workspace project's missing Python dependencies into its local virtual environment "
            "before implementation or validation continues."
        )
        command = build_python_package_install_request(
            thread_id=thread_id,
            packages=packages_to_install,
            rationale=rationale,
        )
        previous_approval = delegated_result.get("approval_request")
        approval_request = {
            "tool_name": "install_python_requirements",
            "command": command,
            "rationale": rationale,
            "user_message": self._render_dependency_approval_message(
                user_language=user_language,
                dependency_check=dependency_check,
                command_preview=command["preview"],
                previous_approval=previous_approval,
            ),
        }
        dependency_check["approval_preview"] = command["preview"]
        if previous_approval is not None:
            dependency_check["deferred_approval_request"] = {
                "tool_name": previous_approval.get("tool_name"),
                "preview": (previous_approval.get("command") or {}).get("preview"),
            }

        reply_parts = [delegated_result.get("reply", "").strip(), approval_request["user_message"].strip()]
        delegated_result["reply"] = "\n\n".join(part for part in reply_parts if part)
        delegated_result["user_reply"] = delegated_result["reply"]
        delegated_result["approval_request"] = approval_request
        delegated_result["approval_request_created"] = True
        delegated_result["status"] = "completed_with_approval" if delegated_result.get("actions_executed") else "approval_required"
        prior_summary = delegated_result.get("internal_summary", "").strip()
        dependency_summary = dependency_check["summary"]
        delegated_result["internal_summary"] = " ".join(
            part for part in (prior_summary, f"Dependency check: {dependency_summary}") if part
        ).strip()
        delegated_result["internal_payload"] = internal_payload
        return delegated_result

    def _dependency_check_summary(
        self,
        *,
        user_language: str,
        missing_requirements: list[str],
        missing_installations: list[str],
        packages_to_install: list[str],
    ) -> str:
        if not packages_to_install:
            return self.language_policy.user_text(
                user_language,
                "Der Dependency-Check hat keine fehlenden Drittanbieter-Pakete gefunden.",
                "The dependency check did not find any missing third-party packages.",
            )

        details: list[str] = []
        if missing_requirements:
            details.append(
                self.language_policy.user_text(
                    user_language,
                    f"Nicht deklarierte Imports: {', '.join(missing_requirements[:6])}.",
                    f"Undeclared imports: {', '.join(missing_requirements[:6])}.",
                )
            )
        if missing_installations:
            details.append(
                self.language_policy.user_text(
                    user_language,
                    f"Bereits deklarierte, aber im Workspace noch nicht installierte Pakete: {', '.join(missing_installations[:6])}.",
                    f"Already declared but not yet installed in the workspace: {', '.join(missing_installations[:6])}.",
                )
            )
        install_part = self.language_policy.user_text(
            user_language,
            f"Vorgeschlagene Installation: {', '.join(packages_to_install[:6])}.",
            f"Proposed installation: {', '.join(packages_to_install[:6])}.",
        )
        return " ".join([*details, install_part]).strip()

    def _dependency_follow_up(
        self,
        *,
        user_language: str,
        packages_to_install: list[str],
    ) -> list[str]:
        if not packages_to_install:
            return []
        return [
            self.language_policy.user_text(
                user_language,
                "Bestätige die Installation der fehlenden Python-Pakete, damit die Implementierung im Workspace-venv weiterlaufen kann.",
                "Approve the missing Python package installation so implementation can continue inside the workspace virtualenv.",
            )
        ]

    def _render_dependency_approval_message(
        self,
        *,
        user_language: str,
        dependency_check: dict,
        command_preview: str,
        previous_approval: dict | None,
    ) -> str:
        missing_requirements = self._dedupe_items(dependency_check.get("missing_requirements") or [])
        missing_installations = self._dedupe_items(dependency_check.get("missing_installations") or [])
        requirements_file = dependency_check.get("workspace_requirements_file", "requirements.txt")
        venv_dir = dependency_check.get("workspace_venv_dir", ".venv")

        lines = [
            self.language_policy.user_text(
                user_language,
                "Freigabe nötig: Der Workspace-Dependency-Check hat fehlende Python-Pakete erkannt.",
                "Approval required: the workspace dependency check found missing Python packages.",
            )
        ]
        if missing_requirements:
            lines.append(
                self.language_policy.user_text(
                    user_language,
                    f"- Neue Pakete aus den Imports: {', '.join(missing_requirements[:8])}",
                    f"- New packages inferred from imports: {', '.join(missing_requirements[:8])}",
                )
            )
        if missing_installations:
            lines.append(
                self.language_policy.user_text(
                    user_language,
                    f"- Bereits in `{requirements_file}` deklarierte, aber lokal noch nicht installierte Pakete: {', '.join(missing_installations[:8])}",
                    f"- Already declared in `{requirements_file}` but not yet installed locally: {', '.join(missing_installations[:8])}",
                )
            )
        lines.extend(
            [
                self.language_policy.user_text(
                    user_language,
                    f"- Geplanter Befehl: `{command_preview}`",
                    f"- Planned command: `{command_preview}`",
                ),
                self.language_policy.user_text(
                    user_language,
                    f"- Zielumgebung: `{venv_dir}` im jeweiligen Workspace",
                    f"- Target environment: `{venv_dir}` inside the workspace",
                ),
                self.language_policy.user_text(
                    user_language,
                    f"- `{requirements_file}` wird bei erfolgreicher Installation mit neuen Direkt-Abhängigkeiten ergänzt.",
                    f"- `{requirements_file}` will be updated with newly approved direct dependencies after a successful install.",
                ),
            ]
        )
        if previous_approval is not None:
            previous_preview = (previous_approval.get("command") or {}).get("preview")
            if previous_preview:
                lines.append(
                    self.language_policy.user_text(
                        user_language,
                        f"- Eine bereits vorbereitete Ausführung (`{previous_preview}`) wird bis nach der Paketinstallation verschoben.",
                        f"- A previously prepared execution (`{previous_preview}`) will be deferred until after the package installation.",
                    )
                )
        return "\n".join(lines)

    def _execute_autonomous_coding_loop(
        self,
        *,
        thread_id: str,
        user_message: str,
        history: list[dict],
        planning_note: dict | None,
        worker_task: str,
        coding_structured_plan: CodingDelegationPlan | None,
        user_language: str,
        consultation: dict | None,
        pending_message_id: int | None = None,
    ) -> dict:
        outline = self._project_outline_dict(self._llm_project_outline(planning_note)) or self._project_outline_from_memory(thread_id)
        if not outline:
            raise ValueError("Autonomous coding loop requested without a project outline.")

        outline["autonomous_execution"] = bool(
            outline.get("autonomous_execution")
            or self._user_requests_full_implementation(user_message)
            or (self._get_project_state(thread_id) or {}).get("autonomous_mode")
        )
        self._store_autonomous_project_plan(thread_id, user_message, outline)
        current_state = self._get_project_state(thread_id) or {}
        completed_steps = list(current_state.get("completed_steps") or [])
        ordered_steps = self._ordered_project_steps(outline)
        latest_result: dict | None = None
        latest_review: dict | None = None
        loop_runs: list[dict] = []
        pending_step: str | None = current_state.get("pending_step")

        if pending_step:
            raise ValueError("Autonomous loop cannot start while a previous step is still awaiting approval.")

        for _ in range(MANAGER_AUTONOMOUS_MAX_STEPS):
            remaining_steps = [step for step in ordered_steps if step not in completed_steps]
            if not remaining_steps:
                break
            current_step = remaining_steps[0]
            step_task = self._build_autonomous_step_task(
                thread_id=thread_id,
                worker_task=worker_task,
                outline=outline,
                current_step=current_step,
                step_index=len(completed_steps),
                total_steps=len(ordered_steps),
                completed_steps=completed_steps,
                remaining_steps=remaining_steps[1:],
            )
            step_result, step_review = self._execute_coding_step(
                thread_id=thread_id,
                user_message=user_message,
                history=history,
                worker_task=step_task,
                coding_structured_plan=coding_structured_plan if not completed_steps else None,
                user_language=user_language,
                consultation=consultation,
                pending_message_id=pending_message_id,
            )
            latest_result = step_result
            latest_review = step_review
            status = step_result.get("status", "")
            if status in {"approval_required", "completed_with_approval"}:
                pending_step = current_step
                self._update_autonomous_project_state(
                    thread_id,
                    user_message=user_message,
                    outline=outline,
                    completed_steps=completed_steps,
                    pending_step=pending_step,
                    latest_status=status,
                    latest_summary=step_result.get("internal_summary", ""),
                    ready_to_test=False,
                    delegated_result=step_result,
                )
                loop_runs.append(
                    self._build_autonomous_run_item(current_step, step_result, step_review)
                )
                break
            if status in {"blocked", "error"}:
                self._update_autonomous_project_state(
                    thread_id,
                    user_message=user_message,
                    outline=outline,
                    completed_steps=completed_steps,
                    pending_step=None,
                    latest_status=status,
                    latest_summary=step_result.get("internal_summary", ""),
                    ready_to_test=False,
                    delegated_result=step_result,
                )
                loop_runs.append(
                    self._build_autonomous_run_item(current_step, step_result, step_review)
                )
                break
            completed_steps.append(current_step)
            ready_to_test = len(completed_steps) >= len(ordered_steps)
            self._update_autonomous_project_state(
                thread_id,
                user_message=user_message,
                outline=outline,
                completed_steps=completed_steps,
                pending_step=None,
                latest_status=status,
                latest_summary=step_result.get("internal_summary", ""),
                ready_to_test=ready_to_test,
                delegated_result=step_result,
            )
            loop_runs.append(
                self._build_autonomous_run_item(current_step, step_result, step_review)
            )
            if ready_to_test:
                break

        if latest_result is None:
            latest_result = {
                "status": "completed",
                "reply": "",
                "user_reply": "",
                "internal_summary": "No autonomous coding step was executed.",
                "actions_executed": [],
                "actions_blocked": [],
                "internal_payload": {"language": "en", "thread_id": thread_id, "task": worker_task},
            }

        ready_to_test = len(completed_steps) >= len(ordered_steps) and pending_step is None
        if ordered_steps and not ready_to_test and pending_step is None and completed_steps:
            project_state = self._get_project_state(thread_id) or {}
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="project_state",
                title="Project State",
                summary=latest_result.get("internal_summary", "") or "Autonomous project run paused after the current step budget.",
                content={
                    **project_state,
                    "phase": "implementation",
                    "awaiting_user_feedback": True,
                    "autonomous_mode": bool(outline.get("autonomous_execution")),
                    "ready_to_test": False,
                    "last_user_request": user_message,
                    "completed_steps": completed_steps,
                    "pending_step": None,
                    "remaining_steps": [step for step in ordered_steps if step not in completed_steps],
                    "next_steps": [step for step in ordered_steps if step not in completed_steps][:3],
                },
            )

        internal_payload = dict(latest_result.get("internal_payload") or {})
        internal_payload["autonomous_run"] = {
            "enabled": True,
            "steps_completed": completed_steps,
            "steps_total": ordered_steps,
            "loop_runs": loop_runs,
            "pending_step": pending_step,
            "ready_to_test": ready_to_test,
        }
        latest_result["internal_payload"] = internal_payload
        latest_result["reply"] = self._render_autonomous_run_reply(
            user_language=user_language,
            loop_runs=loop_runs,
            completed_steps=completed_steps,
            ordered_steps=ordered_steps,
            pending_step=pending_step,
            latest_result=latest_result,
            latest_review=latest_review,
        )
        latest_result["user_reply"] = latest_result["reply"]
        return {
            "delegated_result": latest_result,
            "implementation_review": latest_review,
        }

    def _build_autonomous_run_item(
        self,
        step: str,
        step_result: dict,
        step_review: dict | None,
    ) -> dict:
        internal_payload = step_result.get("internal_payload") or {}
        self_check = internal_payload.get("self_check") or {}
        dependency_check = internal_payload.get("dependency_check") or {}
        review_summary = " ".join(
            self._dedupe_items(
                [
                    (step_review or {}).get("summary", ""),
                    dependency_check.get("summary", ""),
                ]
            )
        ).strip()
        return {
            "step": step,
            "status": step_result.get("status", ""),
            "reply": step_result.get("reply", ""),
            "review_summary": review_summary,
            "touched_paths": self._dedupe_items(self_check.get("touched_paths") or []),
            "follow_up": self._dedupe_items(
                list(self_check.get("follow_up") or []) + list(dependency_check.get("follow_up") or [])
            ),
        }

    def _render_autonomous_run_reply(
        self,
        *,
        user_language: str,
        loop_runs: list[dict],
        completed_steps: list[str],
        ordered_steps: list[str],
        pending_step: str | None,
        latest_result: dict,
        latest_review: dict | None,
    ) -> str:
        ready_to_test = len(completed_steps) >= len(ordered_steps) and pending_step is None
        latest_status = latest_result.get("status", "")
        completed_count = len(completed_steps)
        total_count = len(ordered_steps)
        touched_paths = self._dedupe_items(
            [
                path
                for item in loop_runs
                for path in (item.get("touched_paths") or [])
            ]
        )
        review_summaries = self._dedupe_items(
            [item.get("review_summary", "") for item in loop_runs] + [((latest_review or {}).get("summary", ""))]
        )[:3]
        follow_up = self._dedupe_items(
            [entry for item in loop_runs for entry in (item.get("follow_up") or [])]
        )[:4]

        if ready_to_test:
            intro = self.language_policy.user_text(
                user_language,
                f"Die Implementierung ist aktuell fertig für einen ersten Testlauf. Abgeschlossen: {completed_count}/{total_count} geplante Schritte.",
                f"The implementation is currently ready for a first test run. Completed: {completed_count}/{total_count} planned steps.",
            )
        elif pending_step:
            intro = self.language_policy.user_text(
                user_language,
                f"Die Implementierung ist noch nicht vollständig abgeschlossen. Aktuell sind {completed_count}/{total_count} geplante Schritte erledigt und der nächste Schritt wartet auf Freigabe: {pending_step}",
                f"The implementation is not fully finished yet. {completed_count}/{total_count} planned steps are complete and the next step is waiting for approval: {pending_step}",
            )
        elif latest_status in {"blocked", "error"}:
            intro = self.language_policy.user_text(
                user_language,
                f"Die Implementierung wurde unterbrochen. Abgeschlossen: {completed_count}/{total_count} geplante Schritte.",
                f"The implementation was interrupted. Completed: {completed_count}/{total_count} planned steps.",
            )
        else:
            intro = self.language_policy.user_text(
                user_language,
                f"Die Implementierung läuft weiter. In diesem Lauf wurden {completed_count}/{total_count} geplante Schritte abgeschlossen.",
                f"The implementation is still progressing. This run completed {completed_count}/{total_count} planned steps.",
            )

        lines = [intro]
        if completed_steps:
            title = "Umgesetzte Schritte" if user_language == "de" else "Implemented steps"
            lines.append(self._format_section(title, completed_steps, numbered=True))
        if touched_paths:
            title = "Betroffene Dateien" if user_language == "de" else "Touched files"
            lines.append(self._format_section(title, touched_paths, numbered=False))
        if review_summaries:
            title = "Wichtige Checks" if user_language == "de" else "Important checks"
            lines.append(self._format_section(title, review_summaries, numbered=False))
        if follow_up and not ready_to_test:
            title = "Offene Punkte" if user_language == "de" else "Open items"
            lines.append(self._format_section(title, follow_up, numbered=False))
        return "\n\n".join(line for line in lines if line.strip())

    def _execute_worker_route(
        self,
        thread_id: str,
        user_message: str,
        history: list[dict],
        route: RouteDecision,
        planning_note: dict | None,
        fallback_internal_message: str,
        user_language: str,
        pending_message_id: int | None = None,
    ) -> dict:
        delegated_result = None
        coding_structured_plan = None
        structured_plan_used = False
        fallback_to_heuristic = False
        plan_validation_success = False
        structured_action_count = 0
        consultation = None
        implementation_review = None
        worker_task = self._worker_task(thread_id, user_message, planning_note, fallback_internal_message)

        if route.decision == ManagerDecision.CODING:
            if pending_message_id is not None:
                self._update_pending_status(
                    pending_message_id=pending_message_id,
                    comment=self.language_policy.user_text(
                        user_language,
                        "Coding-Auftrag wird in Arbeitspakete zerlegt",
                        "Breaking the coding task into work packages",
                    ),
                )
            coding_structured_plan, structured_plan_used, fallback_to_heuristic, plan_validation_success = (
                self._resolve_coding_structured_plan(user_message, planning_note)
            )
            structured_action_count = len(coding_structured_plan.actions.actions) if coding_structured_plan is not None else 0
            consultation = self._prepare_coding_consultation(
                thread_id=thread_id,
                user_message=user_message,
                history=history,
                planning_note=planning_note,
                user_language=user_language,
                pending_message_id=pending_message_id,
            )
            if consultation is not None:
                if consultation.get("review_status") not in {"", "completed"}:
                    delegated_result = self._consultation_blocked_result(
                        thread_id=thread_id,
                        user_language=user_language,
                        consultation=consultation,
                    )
                    self._store_delegation_artifacts(
                        thread_id,
                        ManagerDecision.CODING,
                        user_message,
                        delegated_result,
                        consultation,
                        implementation_review=None,
                    )
                    self._update_project_state_after_coding(
                        thread_id=thread_id,
                        user_message=user_message,
                        delegated_result=delegated_result,
                        consultation=consultation,
                        implementation_review=None,
                    )
                    reply = self._manager_wrap(
                        route.reason,
                        delegated_result,
                        planning_note,
                        user_language,
                        user_message,
                        consultation=consultation,
                        implementation_review=None,
                    )
                    return {
                        "delegated_result": delegated_result,
                        "reply": reply,
                        "coding_structured_plan": coding_structured_plan,
                        "structured_plan_used": structured_plan_used,
                        "fallback_to_heuristic": fallback_to_heuristic,
                        "plan_validation_success": plan_validation_success,
                        "structured_action_count": structured_action_count,
                    }
                worker_task = self._augment_coding_task(worker_task, consultation)
        if route.decision == ManagerDecision.CODING and self._should_run_autonomous_coding_loop(
            thread_id,
            user_message,
            planning_note,
        ):
            autonomous_result = self._execute_autonomous_coding_loop(
                thread_id=thread_id,
                user_message=user_message,
                history=history,
                planning_note=planning_note,
                worker_task=worker_task,
                coding_structured_plan=coding_structured_plan,
                user_language=user_language,
                consultation=consultation,
                pending_message_id=pending_message_id,
            )
            delegated_result = autonomous_result["delegated_result"]
            implementation_review = autonomous_result["implementation_review"]
        elif route.decision == ManagerDecision.CODING:
            delegated_result, implementation_review = self._execute_coding_step(
                thread_id=thread_id,
                user_message=user_message,
                history=history,
                worker_task=worker_task,
                coding_structured_plan=coding_structured_plan,
                user_language=user_language,
                consultation=consultation,
                pending_message_id=pending_message_id,
            )
            delegated_result, implementation_review = self._run_manager_repair_loop(
                thread_id=thread_id,
                user_message=user_message,
                history=history,
                worker_task=worker_task,
                user_language=user_language,
                consultation=consultation,
                delegated_result=delegated_result,
                implementation_review=implementation_review,
                pending_message_id=pending_message_id,
            )
            delegated_result = self._enforce_manager_constraints_on_result(
                thread_id=thread_id,
                user_message=user_message,
                delegated_result=delegated_result,
                user_language=user_language,
            )
            self._store_delegation_artifacts(
                thread_id,
                ManagerDecision.CODING,
                user_message,
                delegated_result,
                consultation,
                implementation_review=implementation_review,
            )
            self._update_project_state_after_coding(
                thread_id=thread_id,
                user_message=user_message,
                delegated_result=delegated_result,
                consultation=consultation,
                implementation_review=implementation_review,
            )
        else:
            delegated_result = self.delegation.execute(
                route.decision,
                thread_id,
                user_message,
                history,
                internal_task=worker_task,
                structured_plan=coding_structured_plan,
            )
            self._store_delegation_artifacts(
                thread_id,
                route.decision,
                user_message,
                delegated_result,
                consultation,
                implementation_review=implementation_review,
            )

        if route.decision == ManagerDecision.CODING and delegated_result is not None:
            delegated_result = self._enforce_manager_constraints_on_result(
                thread_id=thread_id,
                user_message=user_message,
                delegated_result=delegated_result,
                user_language=user_language,
            )

        reply = self._manager_wrap(
            route.reason,
            delegated_result,
            planning_note,
            user_language,
            user_message,
            consultation=consultation,
            implementation_review=implementation_review,
        )
        return {
            "delegated_result": delegated_result,
            "reply": reply,
            "coding_structured_plan": coding_structured_plan,
            "structured_plan_used": structured_plan_used,
            "fallback_to_heuristic": fallback_to_heuristic,
            "plan_validation_success": plan_validation_success,
            "structured_action_count": structured_action_count,
        }

    def _consultation_blocked_result(
        self,
        *,
        thread_id: str,
        user_language: str,
        consultation: dict | None,
    ) -> dict:
        review_status = str((consultation or {}).get("review_status", "")).strip() or "error"
        review_summary = str((consultation or {}).get("review_summary", "")).strip() or "Internal review was unavailable."
        reply = self.language_policy.user_text(
            user_language,
            "Ich stoppe an dieser Stelle bewusst, weil die interne Gegenprüfung für den nächsten Coding-Schritt fehlgeschlagen ist. "
            "Ich möchte den Schritt nicht blind ausführen.\n\n"
            f"Interner Review-Status: {review_status}\n"
            f"Interner Hinweis: {review_summary}",
            "I am intentionally pausing here because the internal review for the next coding step failed. "
            "I do not want to execute the step blindly.\n\n"
            f"Internal review status: {review_status}\n"
            f"Internal note: {review_summary}",
        )
        return {
            "status": "blocked",
            "reply": reply,
            "user_reply": reply,
            "tool_results": [],
            "internal_summary": "The manager paused before coding because the internal review step failed.",
            "actions_executed": [],
            "actions_blocked": [],
            "internal_payload": {
                "language": "en",
                "thread_id": thread_id,
                "task": "manager_pause_before_coding",
                "policy": "Do not continue blindly when the internal review for the next coding step failed.",
                "consultation": consultation or {},
            },
        }

    def _plan_safely(self, history_text: str, user_message: str, user_language: str) -> dict | None:
        try:
            planning_note = self.planner.plan(
                history_text=history_text,
                user_message=user_message,
                user_language=user_language,
            )
            if not isinstance(planning_note, dict):
                raise ValueError("Manager planner did not return a valid plan.")
            return planning_note
        except Exception as exc:
            logger.warning("manager_planner_error error=%s", exc)
            return {
                "enabled": False,
                "error": str(exc),
                "error_code": getattr(exc, "code", "manager_planner_error"),
                "status_code": getattr(exc, "status_code", None),
                "model": getattr(self.planner, "model", None),
                "source": "error",
            }

    def _resolve_route(self, thread_id: str, user_message: str, planning_note: dict | None) -> RouteDecision:
        plan = self._llm_plan(planning_note)
        if plan is None:
            raise ValueError("Manager planner did not return a valid plan.")
        decision = ManagerDecision(plan.decision)
        if self._should_rebase_change_request(thread_id, user_message):
            return RouteDecision(
                decision=ManagerDecision.CODING,
                reason="The user is asking for a focused change to the current project state, so the manager should rebase onto the latest implementation instead of resuming stale roadmap steps.",
            )
        if self._should_resume_project_from_feedback(thread_id, user_message):
            return RouteDecision(
                decision=ManagerDecision.CODING,
                reason="The user approved continuing the planned project work, so the next bounded implementation step should start now.",
            )
        if decision != ManagerDecision.PLAN and self._should_force_project_planning(user_message, planning_note):
            return RouteDecision(
                decision=ManagerDecision.PLAN,
                reason="The user asked for planning, decomposition, or feedback before broad implementation starts.",
            )
        return RouteDecision(decision=decision, reason=plan.reason)

    def _llm_decision(self, planning_note: dict | None) -> str | None:
        plan = self._llm_plan(planning_note)
        if plan is None:
            return None
        return plan.decision

    def _has_explicit_workspace_action(self, user_message: str) -> bool:
        lowered = user_message.lower()
        return any(re.search(pattern, lowered) for pattern in EXPLICIT_WORKSPACE_ACTION_PATTERNS)

    def _build_planning_context(self, thread_id: str, messages: list[dict], limit: int = 6) -> str:
        history_text = format_thread_history(messages, limit=limit)
        artifacts = self.store.list_artifacts(thread_id)
        if not artifacts:
            return history_text
        artifact_map = {artifact["kind"]: artifact for artifact in artifacts}
        preferred_order = (
            "project_state",
            "project_plan",
            "manager_constraints",
            "change_request_brief",
            "project_brief",
            "coding_step_contract",
            "coding_change_snapshot",
            "coding_context_snapshot",
            "coding_status",
            "implementation_review",
            "research_notes",
            "review_notes",
            "web_research_notes",
        )
        relevant = []
        for kind in preferred_order:
            artifact = artifact_map.get(kind)
            if artifact is None:
                continue
            summary = artifact.get("summary", "").strip()
            if summary:
                relevant.append(f"{artifact['kind']}: {summary}")
            if len(relevant) >= 5:
                break
        if not relevant:
            return history_text
        constraints = self._get_manager_constraints(thread_id)
        constraint_block = self._manager_constraints_block(constraints, include_header=True)
        if constraint_block:
            return f"{history_text}\n\nProject memory:\n" + "\n".join(relevant) + f"\n\n{constraint_block}"
        return f"{history_text}\n\nProject memory:\n" + "\n".join(relevant)

    def _project_memory_exists(self, thread_id: str) -> bool:
        return bool(self._get_project_state(thread_id) or self._get_project_plan(thread_id) or self._get_coding_status(thread_id))

    def _is_change_request_message(self, user_message: str) -> bool:
        lowered = user_message.lower()
        if any(pattern in lowered for pattern in PLAN_REQUEST_PATTERNS):
            return False
        if any(pattern in lowered for pattern in PROJECT_CONTINUE_PATTERNS):
            return False
        if any(pattern in lowered for pattern in PROJECT_CHANGE_PATTERNS):
            return True
        if any(token in lowered for token in (" only ", " just ", " nur ", " jetzt ", " now ")) and (
            " ai " in f" {lowered} "
            or " gui" in lowered
            or " layout" in lowered
            or "board" in lowered
            or "modus" in lowered
            or "mode" in lowered
        ):
            return True
        return False

    def _should_rebase_change_request(self, thread_id: str, user_message: str) -> bool:
        if not thread_id or not self._project_memory_exists(thread_id):
            return False
        project_state = self._get_project_state(thread_id) or {}
        if self._project_phase(project_state) == "awaiting_approval":
            return False
        return self._is_change_request_message(user_message)

    def _latest_change_paths(self, thread_id: str) -> list[str]:
        artifact = self.store.get_artifact(thread_id, "coding_change_snapshot")
        if artifact is None:
            return []
        content = artifact.get("content") or {}
        if not isinstance(content, dict):
            return []
        return self._dedupe_items(
            [
                str(change.get("path", "")).strip()
                for change in (content.get("snapshot_changes") or [])
                if str(change.get("path", "")).strip() and str(change.get("status", "")).strip().lower() != "unchanged"
            ]
        )[:8]

    def _get_change_request_brief(self, thread_id: str) -> dict:
        artifact = self.store.get_artifact(thread_id, "change_request_brief")
        if artifact is None:
            return {}
        content = artifact.get("content") or {}
        return content if isinstance(content, dict) else {}

    def _store_change_request_brief(self, thread_id: str, user_message: str) -> dict:
        existing = self._get_change_request_brief(thread_id)
        project_state = self._get_project_state(thread_id) or {}
        latest_step_contract = (self.store.get_artifact(thread_id, "coding_step_contract") or {}).get("content", {}) or {}
        latest_review = (self.store.get_artifact(thread_id, "implementation_review") or {}).get("content", {}) or {}
        latest_review_payload = latest_review.get("internal_payload") or {}
        request_kind = "repair_request" if (
            self._project_phase(project_state) == "needs_repair"
            or str(project_state.get("review_verdict", "")).strip().lower() == "needs_repair"
            or any(pattern in user_message.lower() for pattern in PROJECT_REPAIR_PATTERNS)
        ) else "change_request"
        hinted_paths = self._dedupe_items(
            self._latest_change_paths(thread_id)
            + list(latest_step_contract.get("relevant_paths") or [])
            + list(latest_step_contract.get("validation_relevant_paths") or [])
        )[:6]
        brief = {
            "user_message": user_message.strip(),
            "summary": user_message.strip() or existing.get("summary", ""),
            "request_kind": request_kind,
            "previous_phase": self._project_phase(project_state),
            "previous_goal": str(latest_step_contract.get("goal", "")).strip(),
            "latest_review_verdict": str(latest_review_payload.get("verdict", "")).strip().lower(),
            "latest_repair_tasks": self._dedupe_items(latest_review_payload.get("repair_tasks") or [])[:4],
            "previous_remaining_steps": self._dedupe_items(project_state.get("remaining_steps") or [])[:6],
            "hinted_paths": hinted_paths,
            "manager_constraints": self._get_manager_constraints(thread_id),
        }
        self.store.upsert_artifact(
            thread_id=thread_id,
            kind="change_request_brief",
            title="Change Request Brief",
            summary=brief["summary"] or "Focused follow-up change request.",
            content=brief,
        )
        if project_state:
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="project_state",
                title="Project State",
                summary=project_state.get("last_summary", "") or "Project state updated.",
                content={
                    **project_state,
                    "last_user_request": user_message,
                    "active_request_kind": request_kind,
                    "change_request_active": True,
                    "change_request_summary": brief["summary"],
                    "change_request_paths": hinted_paths,
                    "remaining_steps": [f"Focused {request_kind.replace('_', ' ')}: {brief['summary']}"],
                    "next_steps": [f"Focused {request_kind.replace('_', ' ')}: {brief['summary']}"],
                },
            )
        log_event(
            logger,
            "change_request_rebased",
            thread_id=thread_id,
            request_kind=request_kind,
            hinted_paths=hinted_paths,
            previous_phase=brief["previous_phase"],
        )
        return brief

    def _split_user_directives(self, user_message: str) -> list[str]:
        normalized = re.sub(r"[\r\t]+", " ", user_message.strip())
        if not normalized:
            return []
        chunks = re.split(r"(?:\n+|(?<=[.!?])\s+)", normalized)
        directives: list[str] = []
        for raw in chunks:
            chunk = re.sub(r"\s+", " ", raw).strip(" -•\n\r\t")
            if not chunk:
                continue
            lowered = chunk.lower()
            if any(token in lowered for token in DIRECTIVE_SENTENCE_PATTERNS):
                directives.append(chunk)
        return self._dedupe_items(directives)[:8]

    def _extract_manager_constraints(self, user_message: str, existing: dict | None = None) -> dict:
        current = dict(existing or {})
        lowered = user_message.lower()
        directives = self._dedupe_items((current.get("confirmed_directives") or []) + self._split_user_directives(user_message))[:10]

        def _with_override(current_value: bool, forbid_patterns: tuple[str, ...], allow_patterns: tuple[str, ...]) -> bool:
            if any(pattern in lowered for pattern in allow_patterns):
                return False
            if any(pattern in lowered for pattern in forbid_patterns):
                return True
            return current_value

        forbid_dependency_install = _with_override(
            bool(current.get("forbid_dependency_install")),
            DEPENDENCY_INSTALL_FORBID_PATTERNS,
            DEPENDENCY_INSTALL_ALLOW_PATTERNS,
        )
        forbid_pytest = _with_override(
            bool(current.get("forbid_pytest")),
            PYTEST_FORBID_PATTERNS,
            PYTEST_ALLOW_PATTERNS,
        )
        forbid_validation_execution = _with_override(
            bool(current.get("forbid_validation_execution")),
            EXECUTION_FORBID_PATTERNS,
            EXECUTION_ALLOW_PATTERNS,
        )
        stop_after_step = any(pattern in lowered for pattern in STOP_AFTER_STEP_PATTERNS) or bool(
            re.search(r"\bonly step\b", lowered)
        )
        return {
            "confirmed_directives": directives,
            "forbid_dependency_install": forbid_dependency_install,
            "forbid_pytest": forbid_pytest,
            "forbid_validation_execution": forbid_validation_execution,
            "stop_after_step": stop_after_step,
            "last_user_message": user_message.strip(),
        }

    def _constraints_summary(self, constraints: dict) -> str:
        lines = self._manager_constraint_lines(constraints)
        if not lines:
            return "No explicit manager constraints are stored."
        return " ".join(lines[:3])

    def _manager_constraint_lines(self, constraints: dict | None) -> list[str]:
        if not isinstance(constraints, dict):
            return []
        lines = self._dedupe_items(constraints.get("confirmed_directives") or [])
        if constraints.get("forbid_dependency_install"):
            lines.append("Do not install dependencies unless the user explicitly lifts that restriction.")
        if constraints.get("forbid_pytest"):
            lines.append("Do not run pytest until the user explicitly allows it.")
        if constraints.get("forbid_validation_execution"):
            lines.append("Do not request or run validation executions until the user explicitly allows it.")
        if constraints.get("stop_after_step"):
            lines.append("Stop after the current bounded step and report back before continuing automatically.")
        return self._dedupe_items(lines)[:8]

    def _manager_constraints_block(self, constraints: dict | None, *, include_header: bool = False) -> str:
        lines = self._manager_constraint_lines(constraints)
        if not lines:
            return ""
        block = "\n".join(f"- {line}" for line in lines)
        if include_header:
            return "Confirmed user constraints:\n" + block
        return block

    def _get_manager_constraints(self, thread_id: str) -> dict:
        artifact = self.store.get_artifact(thread_id, "manager_constraints")
        if artifact is None:
            return {}
        content = artifact.get("content") or {}
        return content if isinstance(content, dict) else {}

    def _store_manager_constraints(self, thread_id: str, user_message: str) -> dict:
        current = self._get_manager_constraints(thread_id)
        constraints = self._extract_manager_constraints(user_message, current)
        self.store.upsert_artifact(
            thread_id=thread_id,
            kind="manager_constraints",
            title="Manager Constraints",
            summary=self._constraints_summary(constraints),
            content=constraints,
        )
        project_state = self._get_project_state(thread_id) or {}
        if project_state:
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="project_state",
                title="Project State",
                summary=project_state.get("last_summary", "") or "Project state updated.",
                content={
                    **project_state,
                    "manager_constraints": constraints,
                },
            )
        return constraints

    def _llm_plan(self, planning_note: dict | None) -> ManagerPlan | None:
        if not planning_note:
            return None
        plan = planning_note.get("plan")
        if isinstance(plan, ManagerPlan):
            return plan
        return None

    def _llm_project_outline(self, planning_note: dict | None) -> ProjectOutline | None:
        plan = self._llm_plan(planning_note)
        if plan is None:
            return None
        if isinstance(plan.project_outline, ProjectOutline):
            return plan.project_outline
        return None

    def _project_outline_dict(self, outline: ProjectOutline | dict | None) -> dict | None:
        if isinstance(outline, ProjectOutline):
            return outline.model_dump()
        if isinstance(outline, dict):
            return outline
        return None

    def _worker_task(
        self,
        thread_id: str,
        user_message: str,
        planning_note: dict | None,
        fallback_internal_message: str,
    ) -> str:
        if self._should_rebase_change_request(thread_id, user_message):
            return self._project_change_request_worker_task(thread_id, user_message)
        project_follow_up = self._project_follow_up_worker_task(thread_id, user_message)
        if project_follow_up:
            return project_follow_up
        plan = self._llm_plan(planning_note)
        if plan and plan.internal_task_for_worker.strip():
            return plan.internal_task_for_worker.strip()
        return fallback_internal_message

    def _resolve_coding_structured_plan(
        self,
        user_message: str,
        planning_note: dict | None,
    ) -> tuple[CodingDelegationPlan | None, bool, bool, bool]:
        plan = self._llm_plan(planning_note)
        if plan and plan.coding_plan is not None:
            if plan.coding_plan.actions.actions:
                invalid_paths = [
                    str(action.path)
                    for action in plan.coding_plan.actions.actions
                    if getattr(action, "action_type", "") == "create_file"
                    and not str(getattr(action, "content", "") or "").strip()
                    and str(getattr(action, "path", "") or "").rsplit("/", 1)[-1] not in {"__init__.py", ".gitkeep", ".keep"}
                ]
                if invalid_paths:
                    log_event(
                        logger,
                        "manager_structured_plan_rejected",
                        invalid_paths=invalid_paths[:4],
                        reason="placeholder_only_empty_file_write",
                    )
                    return None, False, True, False
                return plan.coding_plan, True, False, True
            raise ValueError("Manager planner returned an empty coding plan.")
        return None, False, False, False

    def _serialize_planning_note(self, planning_note: dict | None) -> dict | None:
        if planning_note is None:
            return None
        serialized = dict(planning_note)
        plan = serialized.get("plan")
        if isinstance(plan, ManagerPlan):
            serialized["plan"] = plan.model_dump()
        return serialized

    def _should_force_manager_english(self, user_message: str) -> bool:
        lowered = user_message.lower()
        markers = (
            "nur noch auf englisch",
            "only in english",
            "english only",
            "bitte auf englisch",
            "please use english",
        )
        return any(marker in lowered for marker in markers)

    def _preferred_manager_language(
        self,
        thread_id: str,
        detected_user_language: str,
        user_message: str,
    ) -> str:
        if self._should_force_manager_english(user_message):
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="manager_preferences",
                title="Manager Preferences",
                summary="Manager should communicate in English for this thread.",
                content={"preferred_user_language": "en"},
            )
            return "en"
        artifact = self.store.get_artifact(thread_id, "manager_preferences")
        if artifact is not None:
            content = artifact.get("content") or {}
            preferred = str(content.get("preferred_user_language", "")).strip().lower()
            if preferred in LanguagePolicy.SUPPORTED_USER_LANGUAGES:
                return preferred
        return detected_user_language

    def _is_plausible_manager_reply(self, reply: str) -> bool:
        lowered = reply.lower().strip()
        if not lowered:
            return False
        return not any(re.search(pattern, lowered) for pattern in ROLE_CONFLICT_PATTERNS)

    def _is_strategic_request(self, user_message: str) -> bool:
        lowered = user_message.lower()
        return any(pattern in lowered for pattern in STRATEGIC_USER_PATTERNS)

    def _get_project_state(self, thread_id: str) -> dict | None:
        artifact = self.store.get_artifact(thread_id, "project_state")
        if artifact is None:
            return None
        content = artifact.get("content") or {}
        if not isinstance(content, dict):
            return None
        return content

    def _get_project_plan(self, thread_id: str) -> dict | None:
        artifact = self.store.get_artifact(thread_id, "project_plan")
        if artifact is None:
            return None
        content = artifact.get("content") or {}
        if not isinstance(content, dict):
            return None
        return content

    def _get_coding_status(self, thread_id: str) -> dict | None:
        artifact = self.store.get_artifact(thread_id, "coding_status")
        if artifact is None:
            return None
        content = artifact.get("content") or {}
        if not isinstance(content, dict):
            return None
        return content

    def _project_phase(self, project_state: dict | None) -> str:
        if not isinstance(project_state, dict):
            return ""
        return str(project_state.get("phase", "")).strip().lower()

    def _derive_validation_state(
        self,
        thread_id: str,
        delegated_result: dict | None,
        *,
        current_state: dict | None = None,
        execution_result: dict | None = None,
    ) -> dict:
        current = current_state or self._get_project_state(thread_id) or {}
        project_plan = self._get_project_plan(thread_id) or {}
        coding_status = self._get_coding_status(thread_id) or {}
        constraints = self._get_manager_constraints(thread_id)
        delegated = delegated_result or {}
        internal_payload = delegated.get("internal_payload") or {}
        self_check = internal_payload.get("self_check") or coding_status.get("self_check") or {}
        follow_up = [str(item).strip() for item in (self_check.get("follow_up") or []) if str(item).strip()]
        follow_up_lower = [item.lower() for item in follow_up]

        actions_executed = delegated.get("actions_executed")
        if not isinstance(actions_executed, list) or not actions_executed:
            actions_executed = current.get("last_actions_executed") or coding_status.get("actions_executed") or []

        stored_execution = execution_result if isinstance(execution_result, dict) else current.get("last_execution_result")
        validation_steps = project_plan.get("validation_steps") or []
        execution_requested = any(item.get("action_type") == "request_execution" for item in actions_executed if isinstance(item, dict))
        validation_pending = delegated.get("status") in {"approval_required", "completed_with_approval"} or any(
            "pending user approval" in item for item in follow_up_lower
        )
        validation_missing = any("no runtime validation has been requested yet" in item for item in follow_up_lower)
        validation_required = bool(validation_steps or execution_requested or validation_pending or validation_missing)

        if isinstance(stored_execution, dict) and stored_execution:
            returncode = stored_execution.get("returncode")
            preview = stored_execution.get("preview") or ""
            command = stored_execution.get("command")
            if isinstance(command, list):
                preview = preview or " ".join(str(part) for part in command)
            if returncode == 0:
                return {
                    "required": bool(validation_required or validation_steps),
                    "status": "passed",
                    "summary": f"Latest validation run succeeded: {preview or 'approved execution'}",
                }
            if returncode is not None:
                stderr = str(stored_execution.get("stderr") or "").strip()
                detail = stderr.splitlines()[-1][:160] if stderr else "approved execution failed"
                return {
                    "required": True,
                    "status": "failed",
                    "summary": f"Latest validation run failed: {detail}",
                }

        if validation_pending:
            return {
                "required": True,
                "status": "pending",
                "summary": "Validation is pending user approval before the step can be considered ready to test.",
            }
        if validation_required and (
            constraints.get("forbid_dependency_install")
            or constraints.get("forbid_pytest")
            or constraints.get("forbid_validation_execution")
        ):
            return {
                "required": True,
                "status": "deferred_by_user",
                "summary": "Validation is currently deferred because the user explicitly paused installs or validation runs.",
            }
        if validation_missing:
            return {
                "required": True,
                "status": "not_requested",
                "summary": "Validation is still missing for the latest implementation step.",
            }
        if validation_steps:
            return {
                "required": True,
                "status": "not_run",
                "summary": "Planned validation steps exist, but no successful execution result has been recorded yet.",
            }
        return {
            "required": False,
            "status": "not_needed",
            "summary": "No explicit validation gate is currently recorded for this step.",
        }

    def _should_resume_project_from_feedback(self, thread_id: str, user_message: str) -> bool:
        project_state = self._get_project_state(thread_id)
        if not project_state or not project_state.get("awaiting_user_feedback"):
            return False
        lowered = user_message.lower()
        if any(pattern in lowered for pattern in PLAN_REQUEST_PATTERNS):
            return False
        if not any(pattern in lowered for pattern in PROJECT_CONTINUE_PATTERNS):
            return False

        phase = self._project_phase(project_state)
        if phase in {"awaiting_approval", "ready_to_test"}:
            return False
        if phase == "blocked":
            return any(pattern in lowered for pattern in PROJECT_REPAIR_PATTERNS)
        return phase in {"planning", "implementation", "needs_repair", "blocked"}

    def _project_change_request_worker_task(self, thread_id: str, user_message: str) -> str:
        brief = self._store_change_request_brief(thread_id, user_message)
        project_plan = self._get_project_plan(thread_id) or {}
        project_state = self._get_project_state(thread_id) or {}
        latest_step_contract = (self.store.get_artifact(thread_id, "coding_step_contract") or {}).get("content", {}) or {}
        latest_change_summary = ((self.store.get_artifact(thread_id, "coding_change_snapshot") or {}).get("summary", "") or "").strip()
        repo_structure = project_plan.get("repo_structure") or project_state.get("repo_structure") or []
        validation_summary = str(project_state.get("validation_summary", "")).strip()
        hinted_paths = brief.get("hinted_paths") or []
        repair_tasks = brief.get("latest_repair_tasks") or []
        carry_forward_note = ""
        if repair_tasks:
            carry_forward_note = f"\nStill-open repair tasks to respect: {repair_tasks}"
        return (
            "Rebase onto the current project reality and implement only the focused follow-up change request below. "
            "Do not continue stale roadmap steps unless they are strictly required for this exact delta. "
            "Preserve the existing structure and keep the scope transparent.\n\n"
            f"Focused user change request: {user_message}\n"
            f"Request kind: {brief.get('request_kind', 'change_request')}\n"
            f"Previous project phase: {brief.get('previous_phase', '')}\n"
            f"Previous manager step goal: {brief.get('previous_goal', '')}\n"
            f"Previous remaining steps to override: {brief.get('previous_remaining_steps', [])}\n"
            f"Hinted relevant paths: {hinted_paths}\n"
            f"Current repo structure: {repo_structure}\n"
            f"Latest change summary: {latest_change_summary or 'No stored change summary yet.'}\n"
            f"Validation summary: {validation_summary or 'No explicit validation summary is stored yet.'}\n"
            f"Confirmed constraints: {self._manager_constraint_lines(brief.get('manager_constraints')) or ['none']}"
            f"{carry_forward_note}\n\n"
            "Deliver only the requested delta, and make sure the after-state shows that the requested change really exists."
        ).strip()

    def _project_follow_up_worker_task(self, thread_id: str, user_message: str) -> str | None:
        if not self._should_resume_project_from_feedback(thread_id, user_message):
            return None
        project_plan = self._get_project_plan(thread_id) or {}
        project_state = self._get_project_state(thread_id) or {}
        next_steps = project_plan.get("steps") or project_plan.get("next_steps") or []
        approach = project_plan.get("approach") or []
        repo_structure = project_plan.get("repo_structure") or []
        summary = project_plan.get("summary", "")
        completed_steps = project_state.get("completed_steps") or []
        remaining_steps = project_state.get("remaining_steps") or []
        latest_status = project_state.get("latest_status", "")
        phase = project_state.get("phase", "")
        ready_to_test = bool(project_state.get("ready_to_test"))
        review_verdict = str(project_state.get("review_verdict", "")).strip().lower()
        repair_tasks = project_state.get("repair_tasks") or []
        project_completion_notes = project_state.get("project_completion_notes") or []
        validation_status = str(project_state.get("validation_status", "")).strip().lower()
        validation_summary = str(project_state.get("validation_summary", "")).strip()
        execution_context = self._execution_debug_context(project_state.get("last_execution_result"))
        repair_note = ""
        if phase == "needs_repair" or repair_tasks:
            repair_note = (
                f"\nOutstanding repair tasks: {repair_tasks}\n"
                "Focus on these remaining issues before expanding scope."
            )
        completion_note = f"\nProject completion notes: {project_completion_notes}" if project_completion_notes else ""
        return (
            "Continue the active project using the previously approved plan. "
            "Implement only the next bounded step, keep the structure coherent, "
            "and leave the project in a state that is ready for the following step.\n\n"
            f"User confirmation: {user_message}\n"
            f"Project summary: {summary}\n"
            f"Project phase: {phase}\n"
            f"Latest project status: {latest_status}\n"
            f"Ready to test: {ready_to_test}\n"
            f"Latest review verdict: {review_verdict}\n"
            f"Validation status: {validation_status or 'unknown'}\n"
            f"Validation summary: {validation_summary or 'No explicit validation summary is stored yet.'}\n"
            f"Suggested repo structure: {repo_structure}\n"
            f"Completed steps: {completed_steps}\n"
            f"Remaining steps: {remaining_steps}\n"
            f"Suggested next steps: {next_steps}\n"
            f"Planned approach: {approach}\n"
            f"{execution_context}"
            f"{repair_note}"
            f"{completion_note}"
        ).strip()

    def _user_requests_full_implementation(self, user_message: str) -> bool:
        lowered = user_message.lower()
        return any(pattern in lowered for pattern in FULL_IMPLEMENTATION_PATTERNS)

    def _should_force_project_planning(self, user_message: str, planning_note: dict | None) -> bool:
        lowered = user_message.lower()
        if any(pattern in lowered for pattern in PLAN_REQUEST_PATTERNS):
            return True
        plan = self._llm_plan(planning_note)
        if plan is not None and plan.decision == "plan":
            return True
        if plan is not None and plan.decision == "coding" and plan.coding_plan is not None:
            actions = getattr(plan.coding_plan.actions, "actions", [])
            if actions:
                return False
        if self._has_explicit_workspace_action(user_message):
            return False
        if self._is_strategic_request(user_message) and sum(token in lowered for token in LARGE_CODING_PATTERNS) >= 1:
            return True
        return len(user_message) >= 240 and sum(token in lowered for token in LARGE_CODING_PATTERNS) >= 2

    def _should_consult_before_coding(
        self,
        thread_id: str,
        user_message: str,
        planning_note: dict | None,
    ) -> bool:
        if self._should_resume_project_from_feedback(thread_id, user_message):
            return False
        project_plan = self._get_project_plan(thread_id) or {}
        project_state = self._get_project_state(thread_id) or {}
        has_structured_project_memory = any(
            project_plan.get(key)
            for key in ("steps", "next_steps", "repo_structure", "project_outline")
        )
        if has_structured_project_memory and not project_state.get("awaiting_user_feedback", False):
            return False
        lowered = user_message.lower()
        if self._is_strategic_request(user_message):
            return True
        if len(user_message) >= 220:
            return True
        if sum(token in lowered for token in LARGE_CODING_PATTERNS) >= 2:
            return True
        plan = self._llm_plan(planning_note)
        if plan is None:
            return False
        return len((plan.summary or "").split()) >= 10

    def _prepare_coding_consultation(
        self,
        thread_id: str,
        user_message: str,
        history: list[dict],
        planning_note: dict | None,
        user_language: str,
        pending_message_id: int | None = None,
    ) -> dict | None:
        if not self._should_consult_before_coding(thread_id, user_message, planning_note):
            return None

        if pending_message_id is not None:
            self._update_pending_status(
                pending_message_id=pending_message_id,
                comment=self.language_policy.user_text(
                    user_language,
                    "Research-Agent zerlegt die Aufgabe",
                    "Research agent is decomposing the task",
                ),
            )
        base_task = self._worker_task(thread_id, user_message, planning_note, user_message)
        constraint_block = self._manager_constraints_block(self._get_manager_constraints(thread_id))
        constraint_intro = f"\nConfirmed user constraints:\n{constraint_block}\n" if constraint_block else ""
        research_task = (
            "Analyze this coding request before implementation. "
            "Break it into small work packages, list assumptions, identify risks, "
            "and recommend the safest first implementation step.\n\n"
            f"Task: {base_task}\n"
            f"{constraint_intro}"
        )
        research_result = self.delegation.execute(
            ManagerDecision.RESEARCH,
            thread_id,
            user_message,
            history,
            internal_task=research_task,
        )

        if pending_message_id is not None:
            self._update_pending_status(
                pending_message_id=pending_message_id,
                comment=self.language_policy.user_text(
                    user_language,
                    "Kritiker-Agent prüft den Plan",
                    "Reviewer agent is checking the plan",
                ),
            )
        review_task = (
            "Review this proposed implementation approach before coding starts. "
            "Focus on gaps, risky scope, missing checks, and whether the first step is small enough.\n\n"
            f"Original task: {base_task}\n\n"
            f"{('Confirmed user constraints:\\n' + constraint_block + '\\n\\n') if constraint_block else ''}"
            f"Research summary: {research_result.get('internal_summary', '')}\n"
            f"Research reply: {research_result.get('reply', '')}"
        )
        review_result = self.delegation.execute(
            ManagerDecision.REVIEW,
            thread_id,
            user_message,
            history,
            internal_task=review_task,
        )
        return {
            "used": True,
            "research_status": research_result.get("status"),
            "research_summary": research_result.get("internal_summary", ""),
            "research_reply": research_result.get("reply", ""),
            "review_status": review_result.get("status"),
            "review_summary": review_result.get("internal_summary", ""),
            "review_reply": review_result.get("reply", ""),
        }

    def _augment_coding_task(self, worker_task: str, consultation: dict) -> str:
        return (
            f"{worker_task}\n\n"
            "Use this internal planning context.\n"
            f"Research summary: {consultation.get('research_summary', '')}\n"
            f"Reviewer summary: {consultation.get('review_summary', '')}\n"
            "Implement only the smallest safe step that directly advances the task."
        ).strip()

    def _prepare_explorer_context(
        self,
        *,
        thread_id: str,
        user_message: str,
        history: list[dict],
        worker_task: str,
        user_language: str,
        pending_message_id: int | None = None,
    ) -> dict | None:
        if pending_message_id is not None:
            self._update_pending_status(
                pending_message_id=pending_message_id,
                comment=self.language_policy.user_text(
                    user_language,
                    "Explorer-Agent analysiert die relevante Repo-Struktur",
                    "Explorer agent is analyzing the relevant repo structure",
                ),
            )

        explorer_task = (
            "Explore the current repository state for the next coding step. "
            "Identify the smallest set of relevant existing files, summarize the current structure, "
            "and recommend a step-local definition of done plus the best validation checks.\n\n"
            f"Target coding step:\n{worker_task}"
        )
        explorer_result = self.delegation.execute(
            ManagerDecision.EXPLORER,
            thread_id,
            user_message,
            history,
            internal_task=explorer_task,
        )
        if explorer_result.get("status") != "completed":
            return None
        payload = explorer_result.get("internal_payload") or {}
        return {
            "summary": explorer_result.get("internal_summary", ""),
            "reply": explorer_result.get("reply", ""),
            "repo_overview": payload.get("repo_overview", ""),
            "relevant_paths": payload.get("relevant_paths", []),
            "validation_relevant_paths": payload.get("validation_relevant_paths", []),
            "suggested_definition_of_done": payload.get("suggested_definition_of_done", []),
            "suggested_checks": payload.get("suggested_checks", []),
            "risks": payload.get("risks", []),
            "snapshots": payload.get("snapshots", []),
        }

    def _narrow_relevant_paths(
        self,
        *,
        thread_id: str,
        explorer_context: dict | None,
        change_request_brief: dict | None,
    ) -> list[str]:
        latest_step_contract = (self.store.get_artifact(thread_id, "coding_step_contract") or {}).get("content", {}) or {}
        prioritized_groups = [
            (change_request_brief or {}).get("hinted_paths", []),
            self._latest_change_paths(thread_id),
            (explorer_context or {}).get("relevant_paths", []),
            latest_step_contract.get("relevant_paths", []),
        ]
        narrowed: list[str] = []
        for group in prioritized_groups:
            for path in group:
                normalized = str(path).strip()
                if not normalized or normalized.casefold() in {item.casefold() for item in narrowed}:
                    continue
                narrowed.append(normalized)
                if len(narrowed) >= 6:
                    return narrowed
        return narrowed

    def _failing_definition_of_done_items(self, latest_review_payload: dict, definition_of_done: list[str]) -> list[str]:
        repair_tasks = self._dedupe_items(latest_review_payload.get("repair_tasks") or [])
        if not repair_tasks:
            return []
        failing_items = [
            item
            for item in definition_of_done
            if any(token in item.casefold() for token in " ".join(repair_tasks).casefold().split())
        ]
        return self._dedupe_items(failing_items or repair_tasks)[:4]

    def _build_coding_step_contract(
        self,
        *,
        thread_id: str,
        user_message: str,
        worker_task: str,
        coding_structured_plan: CodingDelegationPlan | None,
        consultation: dict | None,
        explorer_context: dict | None,
    ) -> dict:
        change_request_brief = self._get_change_request_brief(thread_id)
        latest_review_payload = ((self.store.get_artifact(thread_id, "implementation_review") or {}).get("content", {}) or {}).get("internal_payload", {}) or {}
        request_kind = str(change_request_brief.get("request_kind", "")).strip() or "implementation_step"
        definition_of_done = self._dedupe_items(
            (explorer_context or {}).get("suggested_definition_of_done", [])
            + (explorer_context or {}).get("suggested_checks", [])
        )[:6]
        if not definition_of_done:
            definition_of_done = [
                "Implement only the bounded step requested by the manager.",
                "Keep the existing repository structure coherent.",
                "Surface any unresolved validation risk instead of claiming completion.",
            ]
        if request_kind == "change_request":
            definition_of_done = self._dedupe_items(
                [
                    "Make the requested delta visible in the after-state of the directly affected files.",
                    *definition_of_done,
                ]
            )[:6]
        elif request_kind == "repair_request":
            definition_of_done = self._dedupe_items(
                [
                    "Fix only the remaining failed items without restarting or broadening the feature.",
                    *self._failing_definition_of_done_items(latest_review_payload, definition_of_done),
                    *definition_of_done,
                ]
            )[:6]
        relevant_paths = self._narrow_relevant_paths(
            thread_id=thread_id,
            explorer_context=explorer_context,
            change_request_brief=change_request_brief,
        )
        validation_relevant_paths = self._dedupe_items(
            (explorer_context or {}).get("validation_relevant_paths", [])
            + list((change_request_brief or {}).get("hinted_paths", [])[:2])
        )[:6]

        return {
            "owner": "manager",
            "goal": (coding_structured_plan.summary if coding_structured_plan is not None else "") or worker_task,
            "rationale": (coding_structured_plan.rationale if coding_structured_plan is not None else "") or "",
            "request_kind": request_kind,
            "change_request_summary": str(change_request_brief.get("summary", "")).strip(),
            "definition_of_done": definition_of_done,
            "failing_definition_of_done": self._failing_definition_of_done_items(latest_review_payload, definition_of_done),
            "relevant_paths": relevant_paths,
            "validation_relevant_paths": validation_relevant_paths,
            "repo_overview": (explorer_context or {}).get("repo_overview", ""),
            "before_snapshots": (explorer_context or {}).get("snapshots", [])[:8],
            "consultation_summary": self._dedupe_items(
                [
                    (consultation or {}).get("research_summary", ""),
                    (consultation or {}).get("review_summary", ""),
                ]
            )[:3],
            "user_request": user_message,
        }

    def _augment_coding_task_with_contract(self, worker_task: str, step_contract: dict) -> str:
        relevant_paths = step_contract.get("relevant_paths") or []
        validation_relevant_paths = step_contract.get("validation_relevant_paths") or []
        before_snapshots = step_contract.get("before_snapshots") or []
        dod_items = step_contract.get("definition_of_done") or []
        repo_overview = step_contract.get("repo_overview", "")
        request_kind = step_contract.get("request_kind", "implementation_step")
        change_request_summary = step_contract.get("change_request_summary", "")
        snapshot_lines = []
        for snapshot in before_snapshots[:4]:
            path = str(snapshot.get("path", "")).strip()
            preview = str(snapshot.get("preview", "")).strip()
            if not path:
                continue
            snapshot_lines.append(f"- {path}: {preview[:220] or '[empty]'}")
        relevant_summary = ", ".join(relevant_paths[:8]) if relevant_paths else "none"
        validation_summary = ", ".join(validation_relevant_paths[:8]) if validation_relevant_paths else "none"
        dod_summary = "\n".join(f"- {item}" for item in dod_items[:6]) if dod_items else "- Keep the step coherent."
        before_summary = "\n".join(snapshot_lines) if snapshot_lines else "- No prior file previews captured."
        return (
            f"{worker_task}\n\n"
            "Use this manager-owned step contract.\n"
            f"Goal: {step_contract.get('goal', '')}\n"
            f"Rationale: {step_contract.get('rationale', '')}\n"
            f"Request kind: {request_kind}\n"
            f"Focused change summary: {change_request_summary or 'none'}\n"
            f"Repo overview: {repo_overview}\n"
            f"Relevant existing paths: {relevant_summary}\n"
            f"Relevant validation or entry-point paths: {validation_summary}\n"
            "Definition of done:\n"
            f"{dod_summary}\n"
            "Relevant file state before this step:\n"
            f"{before_summary}\n"
            "Adapt your implementation to this existing structure instead of starting from scratch."
        ).strip()

    def _capture_snapshot_for_paths(self, thread_id: str, paths: list[str]) -> list[dict]:
        snapshots: list[dict] = []
        for path in self._dedupe_items(paths)[:10]:
            try:
                content = file_tools.read_file(thread_id, path)
            except Exception:
                snapshots.append(
                    {
                        "path": path,
                        "exists": False,
                        "preview": "",
                        "sha256": "",
                    }
                )
                continue
            snapshots.append(
                {
                    "path": path,
                    "exists": True,
                    "preview": content[:400],
                    "bytes": len(content.encode("utf-8")),
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
                }
            )
        return snapshots

    def _build_snapshot_changes(self, before_snapshots: list[dict], after_snapshots: list[dict]) -> list[dict]:
        before_map = {str(item.get("path", "")).strip(): item for item in before_snapshots if str(item.get("path", "")).strip()}
        after_map = {str(item.get("path", "")).strip(): item for item in after_snapshots if str(item.get("path", "")).strip()}
        ordered_paths = self._dedupe_items(list(before_map.keys()) + list(after_map.keys()))
        changes: list[dict] = []

        for path in ordered_paths:
            before = before_map.get(path) or {}
            after = after_map.get(path) or {}
            before_exists = bool(before.get("exists", False))
            after_exists = bool(after.get("exists", False))
            before_sha = str(before.get("sha256", "")).strip()
            after_sha = str(after.get("sha256", "")).strip()
            status = "unchanged"
            if not before_exists and after_exists:
                status = "created"
            elif before_exists and not after_exists:
                status = "deleted"
            elif before_exists and after_exists and before_sha and after_sha and before_sha != after_sha:
                status = "modified"

            changes.append(
                {
                    "path": path,
                    "status": status,
                    "before_exists": before_exists,
                    "after_exists": after_exists,
                    "before_sha256": before_sha,
                    "after_sha256": after_sha,
                }
            )
        return changes[:10]

    def _snapshot_change_summary(self, snapshot_changes: list[dict]) -> str:
        created = sum(1 for item in snapshot_changes if item.get("status") == "created")
        modified = sum(1 for item in snapshot_changes if item.get("status") == "modified")
        deleted = sum(1 for item in snapshot_changes if item.get("status") == "deleted")
        unchanged = sum(1 for item in snapshot_changes if item.get("status") == "unchanged")
        parts: list[str] = []
        if created:
            parts.append(f"{created} created")
        if modified:
            parts.append(f"{modified} modified")
        if deleted:
            parts.append(f"{deleted} deleted")
        if unchanged and not parts:
            parts.append(f"{unchanged} unchanged")
        return ", ".join(parts) if parts else "no snapshot changes recorded"

    def _project_outline_from_memory(self, thread_id: str) -> dict | None:
        project_plan = self._get_project_plan(thread_id) or {}
        outline = project_plan.get("project_outline")
        if isinstance(outline, dict) and outline:
            return outline
        steps = project_plan.get("steps") or []
        repo_structure = project_plan.get("repo_structure") or []
        validation_steps = project_plan.get("validation_steps") or []
        completion_criteria = project_plan.get("completion_criteria") or []
        if not any((steps, repo_structure, validation_steps, completion_criteria)):
            return None
        return {
            "summary": project_plan.get("summary", "") or "Stored project outline.",
            "repo_structure": repo_structure,
            "steps": steps,
            "validation_steps": validation_steps,
            "completion_criteria": completion_criteria,
            "autonomous_execution": False,
        }

    def _ordered_project_steps(self, outline: dict | None) -> list[str]:
        if not outline:
            return []
        return self._dedupe_items((outline.get("steps") or []) + (outline.get("validation_steps") or []))

    def _should_run_autonomous_coding_loop(
        self,
        thread_id: str,
        user_message: str,
        planning_note: dict | None,
    ) -> bool:
        outline = self._llm_project_outline(planning_note)
        if outline is not None and outline.autonomous_execution and self._ordered_project_steps(outline.model_dump()):
            return True
        stored_outline = self._project_outline_from_memory(thread_id)
        if not stored_outline:
            return False
        project_state = self._get_project_state(thread_id) or {}
        if self._should_resume_project_from_feedback(thread_id, user_message) and self._user_requests_full_implementation(user_message):
            return bool(self._ordered_project_steps(stored_outline))
        return bool(project_state.get("autonomous_mode") and self._ordered_project_steps(stored_outline))

    def _store_autonomous_project_plan(self, thread_id: str, user_message: str, outline: dict) -> None:
        existing = self._get_project_plan(thread_id) or {}
        steps = self._ordered_project_steps(outline)
        repo_structure = self._dedupe_items(outline.get("repo_structure") or [])
        project_plan = {
            "user_message": user_message,
            "summary": outline.get("summary", "") or existing.get("summary", "") or "Autonomous project plan prepared.",
            "manager_summary": outline.get("summary", "") or existing.get("manager_summary", "") or "Autonomous project plan prepared.",
            "reason": existing.get("reason", "") or "Autonomous implementation requested.",
            "manager_reply_hint": existing.get("manager_reply_hint", ""),
            "approach": existing.get("approach", []),
            "repo_structure": repo_structure,
            "risks": existing.get("risks", []),
            "steps": steps,
            "next_steps": steps[:3],
            "validation_steps": self._dedupe_items(outline.get("validation_steps") or []),
            "completion_criteria": self._dedupe_items(outline.get("completion_criteria") or []),
            "project_outline": outline,
            "research": existing.get("research"),
            "review": existing.get("review"),
            "awaiting_user_feedback": False,
        }
        self.store.upsert_artifact(
            thread_id=thread_id,
            kind="project_plan",
            title="Project Plan",
            summary=project_plan["summary"],
            content=project_plan,
        )
        self.store.upsert_artifact(
            thread_id=thread_id,
            kind="project_brief",
            title="Project Brief",
            summary=project_plan["summary"],
            content={
                "last_user_request": user_message,
                "summary": project_plan["summary"],
                "repo_structure": repo_structure,
                "steps": steps,
                "validation_steps": project_plan["validation_steps"],
                "completion_criteria": project_plan["completion_criteria"],
            },
        )

    def _build_autonomous_step_task(
        self,
        thread_id: str,
        worker_task: str,
        outline: dict,
        current_step: str,
        step_index: int,
        total_steps: int,
        completed_steps: list[str],
        remaining_steps: list[str],
    ) -> str:
        project_state = self._get_project_state(thread_id) or {}
        repo_structure = outline.get("repo_structure") or []
        validation_steps = outline.get("validation_steps") or []
        completion_criteria = outline.get("completion_criteria") or []
        execution_context = self._execution_debug_context(project_state.get("last_execution_result"))
        debug_attempts = int(project_state.get("debug_attempts") or 0)
        return (
            f"{worker_task}\n\n"
            "Autonomous project execution is enabled.\n"
            f"Current step {step_index + 1} of {total_steps}: {current_step}\n"
            f"Completed steps: {completed_steps}\n"
            f"Remaining steps after this one: {remaining_steps}\n"
            f"Suggested repo structure: {repo_structure}\n"
            f"Validation steps: {validation_steps}\n"
            f"Completion criteria: {completion_criteria}\n"
            f"Autonomous debug repair attempts so far: {debug_attempts}\n"
            f"{execution_context}\n"
            "Prefer reading the relevant existing files before writing new ones when integration details matter.\n"
            "Keep module boundaries coherent and leave the repository in a state that is ready for the next step."
        ).strip()

    def _execution_debug_context(self, execution_result: dict | None) -> str:
        if not isinstance(execution_result, dict) or not execution_result:
            return "No recent execution failure context is available."

        command = execution_result.get("command")
        if isinstance(command, list):
            command_preview = " ".join(str(part) for part in command)
        else:
            command_preview = str(command or execution_result.get("preview") or "").strip()
        stderr = str(execution_result.get("stderr") or "").strip()
        stdout = str(execution_result.get("stdout") or "").strip()
        stderr = stderr[:1200]
        stdout = stdout[:600]
        return (
            "Latest execution context:\n"
            f"- Command: {command_preview or 'n/a'}\n"
            f"- Return code: {execution_result.get('returncode')}\n"
            f"- Stdout preview: {stdout or 'n/a'}\n"
            f"- Stderr preview: {stderr or 'n/a'}\n"
            "If this execution failed, use the traceback and import/runtime errors to repair the code before requesting another run."
        )

    def _update_autonomous_project_state(
        self,
        thread_id: str,
        *,
        user_message: str,
        outline: dict,
        completed_steps: list[str],
        pending_step: str | None,
        latest_status: str,
        latest_summary: str,
        ready_to_test: bool,
        delegated_result: dict,
    ) -> None:
        ordered_steps = self._ordered_project_steps(outline)
        remaining_steps = [step for step in ordered_steps if step not in completed_steps and step != pending_step]
        phase = "implementation"
        awaiting_user_feedback = False
        if pending_step:
            phase = "awaiting_approval"
        elif latest_status in {"blocked", "error"}:
            phase = "blocked"
            awaiting_user_feedback = True
        elif ready_to_test:
            phase = "ready_to_test"
            awaiting_user_feedback = True
        self.store.upsert_artifact(
            thread_id=thread_id,
            kind="project_state",
            title="Project State",
            summary=latest_summary or "Autonomous project run updated.",
            content={
                "phase": phase,
                "awaiting_user_feedback": awaiting_user_feedback,
                "autonomous_mode": bool(outline.get("autonomous_execution")),
                "ready_to_test": ready_to_test,
                "last_user_request": user_message,
                "last_summary": latest_summary,
                "latest_status": latest_status,
                "repo_structure": outline.get("repo_structure", []),
                "steps": ordered_steps,
                "completed_steps": completed_steps,
                "pending_step": pending_step,
                "remaining_steps": remaining_steps,
                "next_steps": remaining_steps[:3],
                "validation_steps": outline.get("validation_steps", []),
                "completion_criteria": outline.get("completion_criteria", []),
                "debug_attempts": int((self._get_project_state(thread_id) or {}).get("debug_attempts") or 0),
                "last_actions_executed": delegated_result.get("actions_executed", []),
                "last_actions_blocked": delegated_result.get("actions_blocked", []),
            },
        )

    def _dedupe_items(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned

    def _format_section(self, title: str, items: list[str], numbered: bool) -> str:
        marker = (lambda idx: f"{idx + 1}.") if numbered else (lambda idx: "-")
        return "\n".join([f"{title}:"] + [f"{marker(idx)} {item}" for idx, item in enumerate(items[:4])])

    def _classify_step_outcome(self, delegated_result: dict, implementation_review: dict | None = None) -> str:
        actions_executed = delegated_result.get("actions_executed") or []
        actions_blocked = delegated_result.get("actions_blocked") or []
        action_types = {str(item.get("action_type", "")).strip() for item in actions_executed if isinstance(item, dict)}
        if delegated_result.get("status") in {"blocked", "error"} or actions_blocked:
            return "blocked"
        internal_payload = delegated_result.get("internal_payload") or {}
        self_check = internal_payload.get("self_check") or {}
        inspected_files = self_check.get("inspected_files") or []
        created_targets = {
            str(item.get("target", "")).strip()
            for item in actions_executed
            if isinstance(item, dict) and str(item.get("action_type", "")).strip() == "create_file"
        }
        empty_created_targets = {
            str(item.get("path", "")).strip()
            for item in inspected_files
            if str(item.get("path", "")).strip() in created_targets and int(item.get("bytes") or 0) == 0
        }
        if action_types.intersection({"create_file", "make_directory", "delete_path"}):
            if created_targets and created_targets == empty_created_targets and action_types.issubset({"create_file"}):
                return "no_effect"
            return "implemented"
        review_payload = (implementation_review or {}).get("internal_payload") or {}
        if any(str(item.get("status", "")).strip().lower() in {"created", "modified", "deleted"} for item in (review_payload.get("snapshot_changes") or [])):
            return "implemented"
        if "request_execution" in action_types:
            return "validation_requested"
        if action_types.intersection({"read_file", "list_files"}) or (self_check.get("inspected_files") or []):
            return "analysis_only"
        return "no_effect"

    def _requested_change_has_after_state_evidence(self, step_contract: dict, review_payload: dict, step_outcome: str) -> bool:
        if step_outcome == "implemented":
            changed_paths = {
                str(item.get("path", "")).strip()
                for item in (review_payload.get("snapshot_changes") or [])
                if str(item.get("status", "")).strip().lower() in {"created", "modified", "deleted"}
            }
            relevant_paths = {
                str(path).strip()
                for path in (step_contract.get("relevant_paths") or [])
                if str(path).strip()
            }
            if not relevant_paths:
                return bool(changed_paths)
            return bool(changed_paths.intersection(relevant_paths))
        return False

    def _apply_review_evidence_gate(self, delegated_result: dict, implementation_review: dict | None) -> dict | None:
        if implementation_review is None:
            return None
        review_payload = dict(implementation_review.get("internal_payload") or {})
        step_contract = ((delegated_result.get("internal_payload") or {}).get("step_contract") or {})
        request_kind = str(step_contract.get("request_kind", "implementation_step")).strip()
        step_outcome = self._classify_step_outcome(delegated_result, implementation_review)
        review_payload["step_outcome"] = step_outcome

        verdict = str(review_payload.get("verdict", "")).strip().lower()
        needs_write_evidence = request_kind in {"change_request", "repair_request", "implementation_step"}
        has_after_state_evidence = self._requested_change_has_after_state_evidence(step_contract, review_payload, step_outcome)

        if needs_write_evidence and step_outcome in {"analysis_only", "no_effect"} and verdict == "done":
            review_payload["verdict"] = "needs_repair"
            review_payload["definition_of_done_met"] = False
            review_payload["project_status"] = "in_progress"
            findings = self._dedupe_items(
                ["The coding step only inspected the workspace and did not apply the requested implementation change."] + list(review_payload.get("findings") or [])
            )
            repair_tasks = self._dedupe_items(
                ["Apply the requested code change in the relevant files instead of stopping after read-only analysis."] + list(review_payload.get("repair_tasks") or [])
            )
            review_payload["findings"] = findings[:6]
            review_payload["repair_tasks"] = repair_tasks[:6]
        elif request_kind in {"change_request", "repair_request"} and verdict == "done" and not has_after_state_evidence:
            review_payload["verdict"] = "needs_repair"
            review_payload["definition_of_done_met"] = False
            review_payload["project_status"] = "in_progress"
            findings = self._dedupe_items(
                ["The requested follow-up change is not visible in the after-state of the relevant files."] + list(review_payload.get("findings") or [])
            )
            repair_tasks = self._dedupe_items(
                ["Implement the requested delta so the after-state of the directly affected files shows the change."] + list(review_payload.get("repair_tasks") or [])
            )
            review_payload["findings"] = findings[:6]
            review_payload["repair_tasks"] = repair_tasks[:6]

        summary = str(implementation_review.get("summary", "")).strip()
        if review_payload.get("verdict") == "needs_repair" and step_outcome in {"analysis_only", "no_effect"}:
            summary = "The step was analyzed, but the requested implementation change did not happen yet."
        elif review_payload.get("verdict") == "needs_repair" and request_kind in {"change_request", "repair_request"} and not has_after_state_evidence:
            summary = "The requested follow-up change is still missing from the visible after-state."

        return {
            **implementation_review,
            "summary": summary or implementation_review.get("summary", ""),
            "internal_payload": review_payload,
        }

    def _should_review_coding_result(self, delegated_result: dict) -> bool:
        status = delegated_result.get("status")
        if status not in {"completed", "completed_with_approval", "approval_required", "blocked"}:
            return False
        actions_executed = delegated_result.get("actions_executed") or []
        actions_blocked = delegated_result.get("actions_blocked") or []
        if not actions_executed and not actions_blocked:
            return False
        internal_payload = delegated_result.get("internal_payload") or {}
        self_check = internal_payload.get("self_check") or {}
        inspected_files = self_check.get("inspected_files") or []
        substantive_preview = any(str(item.get("preview", "")).strip() for item in inspected_files)
        substantive_actions = [item for item in actions_executed if item.get("action_type") != "request_execution"]
        if not substantive_actions and not actions_blocked:
            return False
        return True

    def _review_coding_result(
        self,
        thread_id: str,
        user_message: str,
        history: list[dict],
        delegated_result: dict,
        user_language: str,
        pending_message_id: int | None = None,
    ) -> dict | None:
        if not self._should_review_coding_result(delegated_result):
            return None

        if pending_message_id is not None:
            self._update_pending_status(
                pending_message_id=pending_message_id,
                comment=self.language_policy.user_text(
                    user_language,
                    "Kritiker-Agent prüft den letzten Coding-Schritt",
                    "Reviewer agent is checking the latest coding step",
                ),
            )

        internal_payload = delegated_result.get("internal_payload") or {}
        self_check = internal_payload.get("self_check") or {}
        step_contract = internal_payload.get("step_contract") or {}
        explorer_context = internal_payload.get("explorer_context") or {}
        touched_paths = self._dedupe_items(self_check.get("touched_paths") or [])
        inspected_paths = self._dedupe_items(
            [str(item.get("path", "")).strip() for item in (self_check.get("inspected_files") or [])]
        )
        actions_executed = delegated_result.get("actions_executed") or []
        actions_blocked = delegated_result.get("actions_blocked") or []
        executed_action_types = self._dedupe_items(
            [str(item.get("action_type", "")).strip() for item in actions_executed]
        )
        blocked_action_types = self._dedupe_items(
            [str(item.get("action_type", "")).strip() for item in actions_blocked]
        )
        primary_risks = self._dedupe_items(self_check.get("follow_up") or [])
        validation_state = self._derive_validation_state(thread_id, delegated_result)
        contract_definition_of_done = self._dedupe_items(step_contract.get("definition_of_done") or [])
        contract_relevant_paths = self._dedupe_items(step_contract.get("relevant_paths") or [])
        contract_validation_paths = self._dedupe_items(step_contract.get("validation_relevant_paths") or [])
        before_snapshots = step_contract.get("before_snapshots") or explorer_context.get("snapshots") or []
        review_paths = self._dedupe_items(contract_relevant_paths + contract_validation_paths + touched_paths + inspected_paths)
        after_snapshots = self._capture_snapshot_for_paths(thread_id, review_paths)
        snapshot_changes = self._build_snapshot_changes(before_snapshots, after_snapshots)
        touched_summary = ", ".join(touched_paths[:6]) if touched_paths else "No touched paths recorded."
        inspected_summary = ", ".join(inspected_paths[:6]) if inspected_paths else "No inspected files recorded."
        inspected_previews = []
        for item in (self_check.get("inspected_files") or [])[:4]:
            path = str(item.get("path", "")).strip()
            preview = str(item.get("preview", "")).strip()
            if not path:
                continue
            inspected_previews.append(f"File: {path}\nPreview: {preview or '[empty]'}")
        inspected_preview_block = "\n\n".join(inspected_previews) if inspected_previews else "No file previews were captured."
        before_snapshot_lines = []
        for snapshot in before_snapshots[:6]:
            path = str(snapshot.get("path", "")).strip()
            preview = str(snapshot.get("preview", "")).strip()
            exists = bool(snapshot.get("exists", True))
            if not path:
                continue
            before_snapshot_lines.append(
                f"Before: {path}\nExists: {exists}\nPreview: {preview or '[empty]'}"
            )
        before_snapshot_block = "\n\n".join(before_snapshot_lines) if before_snapshot_lines else "No before snapshots were captured."
        after_snapshot_lines = []
        for snapshot in after_snapshots[:6]:
            path = str(snapshot.get("path", "")).strip()
            preview = str(snapshot.get("preview", "")).strip()
            exists = bool(snapshot.get("exists", False))
            if not path:
                continue
            after_snapshot_lines.append(
                f"After: {path}\nExists: {exists}\nPreview: {preview or '[empty]'}"
            )
        after_snapshot_block = "\n\n".join(after_snapshot_lines) if after_snapshot_lines else "No after snapshots were captured."
        snapshot_change_block = "\n".join(
            f"- {item['path']}: {item['status']}"
            for item in snapshot_changes
            if item.get("status") != "unchanged"
        ) or "No material snapshot changes were detected."
        executed_summary = ", ".join(executed_action_types[:6]) if executed_action_types else "none"
        blocked_summary = ", ".join(blocked_action_types[:6]) if blocked_action_types else "none"
        constraint_block = self._manager_constraints_block(self._get_manager_constraints(thread_id))
        constraint_intro = f"Confirmed user constraints:\n{constraint_block}\n" if constraint_block else ""
        review_task = (
            "Review the latest coding step as a critical implementation reviewer. "
            "Use the manager-owned step contract, the before/after file state, and the implementation summary below to determine "
            "whether the step is actually done, whether the definition of done is met, and which concrete repair tasks remain if it is not. "
            "Also distinguish clearly between bounded step completion and the broader project status.\n\n"
            f"Original user request: {user_message}\n"
            f"{constraint_intro}"
            f"Manager step goal: {step_contract.get('goal', '')}\n"
            f"Manager step rationale: {step_contract.get('rationale', '')}\n"
            f"Relevant existing paths: {', '.join(contract_relevant_paths[:8]) if contract_relevant_paths else 'none'}\n"
            f"Relevant validation or entry-point paths: {', '.join(contract_validation_paths[:8]) if contract_validation_paths else 'none'}\n"
            f"Step definition of done: {contract_definition_of_done[:6] if contract_definition_of_done else 'none'}\n"
            f"Coding status: {delegated_result.get('status')}\n"
            f"Coding summary: {delegated_result.get('internal_summary', '')}\n"
            f"Self-check summary: {self_check.get('summary', '')}\n"
            f"Executed action types: {executed_summary}\n"
            f"Blocked action types: {blocked_summary}\n"
            f"Touched paths: {touched_summary}\n"
            f"Inspected files: {inspected_summary}\n"
            f"Self-check previews:\n{inspected_preview_block}\n"
            f"Relevant file state before the step:\n{before_snapshot_block}\n"
            f"Relevant file state after the step:\n{after_snapshot_block}\n"
            f"Observed snapshot changes:\n{snapshot_change_block}\n"
            f"Known follow-up risks: {primary_risks[:4] if primary_risks else 'none'}\n\n"
            f"Manager validation required: {validation_state['required']}\n"
            f"Manager validation status: {validation_state['status']}\n"
            f"Manager validation summary: {validation_state['summary']}\n\n"
            "Review goals:\n"
            "- State whether the definition of done is met.\n"
            "- Call out the highest-risk gap in the step.\n"
            "- Recommend concrete repair tasks if the step is not done.\n"
            "- State whether the overall project is in progress, ready for validation, validated and ready to test, or actually complete.\n"
            "- Say whether additional validation is needed before expanding scope.\n"
            "- If validation has not passed yet, prefer `ready_for_validation` over `validated_ready_to_test`.\n"
            "- Highlight obvious import, runtime, or integration problems if the previews strongly suggest them.\n"
            "- Stay concise and actionable."
        )
        log_event(
            logger,
            "post_coding_review_started",
            thread_id=thread_id,
            coding_status=delegated_result.get("status"),
            inspected_file_count=len(self_check.get("inspected_files") or []),
            touched_path_count=len(touched_paths),
            touched_paths=touched_paths[:6],
            inspected_paths=inspected_paths[:6],
            follow_up_count=len(primary_risks),
            review_task_chars=len(review_task),
            actions_executed_count=len(actions_executed),
            actions_blocked_count=len(actions_blocked),
        )
        if REVIEWER_DEBUG_LOG_PROMPTS:
            log_text_block(
                logger,
                "post_coding_review_task",
                review_task,
                max_chars=REVIEWER_DEBUG_LOG_MAX_CHARS,
                thread_id=thread_id,
                coding_status=delegated_result.get("status"),
                review_task_chars=len(review_task),
            )
        if POST_CODING_REVIEW_DELAY_SECONDS > 0:
            log_event(
                logger,
                "post_coding_review_delay",
                thread_id=thread_id,
                delay_seconds=POST_CODING_REVIEW_DELAY_SECONDS,
            )
            sleep(POST_CODING_REVIEW_DELAY_SECONDS)
        started = monotonic()
        log_event(
            logger,
            "post_coding_review_delegation_started",
            thread_id=thread_id,
            coding_status=delegated_result.get("status"),
            review_task_chars=len(review_task),
        )
        review_result = self.delegation.execute(
            ManagerDecision.REVIEW,
            thread_id,
            user_message,
            history,
            internal_task=review_task,
        )
        log_event(
            logger,
            "post_coding_review_delegation_returned",
            thread_id=thread_id,
            status=review_result.get("status"),
            reply_chars=len(review_result.get("reply", "") or ""),
            summary_chars=len(review_result.get("internal_summary", "") or ""),
        )
        log_event(
            logger,
            "post_coding_review_completed",
            thread_id=thread_id,
            status=review_result.get("status"),
            elapsed_ms=int((monotonic() - started) * 1000),
            summary_chars=len(review_result.get("internal_summary", "") or ""),
        )
        return {
            "summary": review_result.get("internal_summary", ""),
            "reply": review_result.get("reply", ""),
            "status": review_result.get("status"),
            "internal_payload": {
                **(review_result.get("internal_payload") or {}),
                "before_snapshots": before_snapshots[:8],
                "after_snapshots": after_snapshots[:8],
                "snapshot_changes": snapshot_changes,
                "snapshot_change_summary": self._snapshot_change_summary(snapshot_changes),
            },
        }

    def _store_delegation_artifacts(
        self,
        thread_id: str,
        decision: ManagerDecision,
        user_message: str,
        delegated_result: dict,
        consultation: dict | None = None,
        implementation_review: dict | None = None,
    ) -> None:
        if consultation:
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="research_notes",
                title="Research Notes",
                summary=consultation.get("research_summary", "") or "Internal research notes.",
                content={
                    "user_message": user_message,
                    "reply": consultation.get("research_reply", ""),
                    "summary": consultation.get("research_summary", ""),
                },
            )
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="review_notes",
                title="Review Notes",
                summary=consultation.get("review_summary", "") or "Internal review notes.",
                content={
                    "user_message": user_message,
                    "reply": consultation.get("review_reply", ""),
                    "summary": consultation.get("review_summary", ""),
                },
            )
            project_summary = consultation.get("research_summary", "") or consultation.get("review_summary", "")
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="project_brief",
                title="Project Brief",
                summary=project_summary or "Working project brief.",
                content={
                    "last_user_request": user_message,
                    "research_summary": consultation.get("research_summary", ""),
                    "review_summary": consultation.get("review_summary", ""),
                    "route": decision.value,
                },
            )

        internal_payload = delegated_result.get("internal_payload") or {}
        internal_summary = delegated_result.get("internal_summary", "") or ""
        if decision == ManagerDecision.CODING:
            dependency_check = internal_payload.get("dependency_check") or {}
            step_contract = internal_payload.get("step_contract") or {}
            explorer_context = internal_payload.get("explorer_context") or {}
            step_outcome = str(internal_payload.get("step_outcome", "")).strip() or self._classify_step_outcome(delegated_result, implementation_review)
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="coding_status",
                title="Coding Status",
                summary=internal_summary or "Latest coding step executed.",
                content={
                    "user_message": user_message,
                    "status": delegated_result.get("status"),
                    "actions_executed": delegated_result.get("actions_executed", []),
                    "actions_blocked": delegated_result.get("actions_blocked", []),
                    "workspace": delegated_result.get("workspace"),
                    "self_check": internal_payload.get("self_check", {}),
                    "dependency_check": dependency_check,
                    "step_outcome": step_outcome,
                },
            )
            if step_contract:
                self.store.upsert_artifact(
                    thread_id=thread_id,
                    kind="coding_step_contract",
                    title="Coding Step Contract",
                    summary=step_contract.get("goal", "") or "Latest manager-owned coding step contract.",
                    content=step_contract,
                )
            if explorer_context:
                self.store.upsert_artifact(
                    thread_id=thread_id,
                    kind="coding_context_snapshot",
                    title="Coding Context Snapshot",
                    summary=explorer_context.get("summary", "") or "Latest explorer snapshot before coding.",
                    content=explorer_context,
                )
            if implementation_review is not None:
                review_payload = implementation_review.get("internal_payload") or {}
                self.store.upsert_artifact(
                    thread_id=thread_id,
                    kind="coding_change_snapshot",
                    title="Coding Change Snapshot",
                    summary=review_payload.get("snapshot_change_summary", "") or "Latest before/after coding snapshot.",
                    content={
                        "user_message": user_message,
                        "step_goal": step_contract.get("goal", "") if step_contract else "",
                        "summary": review_payload.get("snapshot_change_summary", ""),
                        "review_verdict": review_payload.get("verdict", ""),
                        "step_outcome": step_outcome,
                        "before_snapshots": review_payload.get("before_snapshots", []),
                        "after_snapshots": review_payload.get("after_snapshots", []),
                        "snapshot_changes": review_payload.get("snapshot_changes", []),
                    },
                )
            if dependency_check:
                self.store.upsert_artifact(
                    thread_id=thread_id,
                    kind="dependency_status",
                    title="Dependency Status",
                    summary=dependency_check.get("summary", "") or "Latest dependency audit.",
                    content=dependency_check,
                )
            if implementation_review is not None:
                self.store.upsert_artifact(
                    thread_id=thread_id,
                    kind="implementation_review",
                    title="Implementation Review",
                    summary=implementation_review.get("summary", "") or "Latest implementation review.",
                    content={
                        "user_message": user_message,
                        "status": implementation_review.get("status"),
                        "reply": implementation_review.get("reply", ""),
                        "summary": implementation_review.get("summary", ""),
                        "project_status": (implementation_review.get("internal_payload") or {}).get("project_status"),
                        "internal_payload": implementation_review.get("internal_payload", {}),
                    },
                )
            return
        if decision == ManagerDecision.RESEARCH:
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="research_notes",
                title="Research Notes",
                summary=internal_summary or "Latest research result.",
                content={
                    "user_message": user_message,
                    "status": delegated_result.get("status"),
                    "reply": delegated_result.get("reply", ""),
                    "internal_payload": internal_payload,
                },
            )
            sources = internal_payload.get("sources") or []
            if sources:
                self.store.upsert_artifact(
                    thread_id=thread_id,
                    kind="web_research_notes",
                    title="Web Research Notes",
                    summary=internal_summary or "Latest web-backed research result.",
                    content={
                        "user_message": user_message,
                        "status": delegated_result.get("status"),
                        "reply": delegated_result.get("reply", ""),
                        "sources": sources,
                        "open_questions": internal_payload.get("open_questions", []),
                        "summary": internal_payload.get("summary", ""),
                    },
                )
            return
        if decision == ManagerDecision.REVIEW:
            self.store.upsert_artifact(
                thread_id=thread_id,
                kind="review_notes",
                title="Review Notes",
                summary=internal_summary or "Latest review result.",
                content={
                    "user_message": user_message,
                    "status": delegated_result.get("status"),
                    "reply": delegated_result.get("reply", ""),
                    "internal_payload": internal_payload,
                },
            )

    def _update_project_state_after_coding(
        self,
        thread_id: str,
        user_message: str,
        delegated_result: dict,
        consultation: dict | None = None,
        implementation_review: dict | None = None,
    ) -> None:
        status = delegated_result.get("status")
        current = self._get_project_state(thread_id) or {}
        manager_constraints = self._get_manager_constraints(thread_id)
        review_payload = (implementation_review or {}).get("internal_payload") or {}
        step_outcome = str((delegated_result.get("internal_payload") or {}).get("step_outcome", "")).strip() or self._classify_step_outcome(delegated_result, implementation_review)

        project_status = str(review_payload.get("project_status", "")).strip().lower()
        review_verdict = str(review_payload.get("verdict", "")).strip().lower()
        definition_of_done_met = review_payload.get("definition_of_done_met")
        repair_tasks = self._dedupe_items(review_payload.get("repair_tasks") or [])[:4]
        project_completion_notes = self._dedupe_items(review_payload.get("project_completion_notes") or [])[:4]
        validation_state = self._derive_validation_state(thread_id, delegated_result, current_state=current)
        validation_required = bool(validation_state["required"])
        validation_status = validation_state["status"]
        validation_summary = validation_state["summary"]
        phase = "implementation"
        awaiting_user_feedback = True
        if status in {"approval_required", "completed_with_approval"}:
            phase = "awaiting_approval"
            awaiting_user_feedback = False
        elif status in {"blocked", "error"} or review_verdict == "blocked":
            phase = "blocked"
            awaiting_user_feedback = True
        elif review_verdict == "needs_repair" or definition_of_done_met is False or repair_tasks or step_outcome in {"analysis_only", "no_effect"}:
            phase = "needs_repair"
            awaiting_user_feedback = True

        next_steps = self._dedupe_items(
            (
                ["Implement the requested change in the relevant files before calling this step complete."]
                if step_outcome in {"analysis_only", "no_effect"}
                else []
            )
            + repair_tasks
            + project_completion_notes
            + ([validation_summary] if validation_required and validation_status != "passed" else [])
            + [
                (implementation_review or {}).get("summary", ""),
                (consultation or {}).get("review_summary", ""),
                (consultation or {}).get("research_summary", ""),
            ]
        )[:3]
        summary = delegated_result.get("internal_summary", "") or current.get("last_summary", "") or "Latest coding step processed."
        ready_to_test = bool(
            step_outcome == "implemented"
            and
            project_status in {"ready_to_test", "validated_ready_to_test", "project_done"}
            and status == "completed"
            and review_verdict not in {"blocked", "needs_repair"}
            and not repair_tasks
            and definition_of_done_met is not False
            and (not validation_required or validation_status == "passed")
        )
        if ready_to_test:
            phase = "ready_to_test"
            awaiting_user_feedback = True
        self.store.upsert_artifact(
            thread_id=thread_id,
            kind="project_state",
            title="Project State",
            summary=summary,
            content={
                "phase": phase,
                "awaiting_user_feedback": awaiting_user_feedback,
                "autonomous_mode": False,
                "ready_to_test": ready_to_test,
                "last_user_request": user_message,
                "last_summary": summary,
                "latest_status": status,
                "review_project_status": project_status,
                "review_verdict": review_verdict,
                "step_outcome": step_outcome,
                "definition_of_done_met": definition_of_done_met,
                "repair_tasks": repair_tasks,
                "project_completion_notes": project_completion_notes,
                "validation_required": validation_required,
                "validation_status": validation_status,
                "validation_summary": validation_summary,
                "next_steps": next_steps,
                "remaining_steps": next_steps,
                "completed_steps": [],
                "pending_step": None,
                "last_actions_executed": delegated_result.get("actions_executed", []),
                "last_actions_blocked": delegated_result.get("actions_blocked", []),
                "last_execution_result": current.get("last_execution_result"),
                "manager_constraints": manager_constraints,
            },
        )

    def _update_project_state_after_execution_approval(self, thread_id: str, result: dict) -> None:
        current = self._get_project_state(thread_id)
        if not current or current.get("autonomous_mode"):
            return

        coding_status = self._get_coding_status(thread_id) or {}
        delegated_result = {
            "status": current.get("latest_status"),
            "actions_executed": current.get("last_actions_executed", []),
            "actions_blocked": current.get("last_actions_blocked", []),
            "internal_payload": {
                "self_check": coding_status.get("self_check", {}),
            },
        }
        validation_state = self._derive_validation_state(
            thread_id,
            delegated_result,
            current_state=current,
            execution_result=result,
        )
        validation_required = bool(validation_state["required"])
        validation_status = validation_state["status"]
        validation_summary = validation_state["summary"]

        if result.get("returncode") == 0:
            ready_to_test = bool(
                current.get("review_project_status") in {"ready_to_test", "validated_ready_to_test", "project_done"}
                and current.get("review_verdict") not in {"blocked", "needs_repair"}
                and not (current.get("repair_tasks") or [])
                and (not validation_required or validation_status == "passed")
            )
            next_steps = self._dedupe_items(
                (current.get("project_completion_notes") or [])
                + ([] if ready_to_test else ["Continue with the next bounded implementation or validation step."])
            )[:3]
            phase = "ready_to_test" if ready_to_test else "implementation"
            latest_status = "executed"
            summary = "Approved execution completed successfully."
        else:
            ready_to_test = False
            next_steps = self._dedupe_items(
                [validation_summary] + (current.get("repair_tasks") or []) + (current.get("project_completion_notes") or [])
            )[:3]
            phase = "blocked"
            latest_status = "failed"
            summary = "Approved execution failed."

        self.store.upsert_artifact(
            thread_id=thread_id,
            kind="project_state",
            title="Project State",
            summary=summary,
            content={
                **current,
                "phase": phase,
                "awaiting_user_feedback": True,
                "ready_to_test": ready_to_test,
                "latest_status": latest_status,
                "validation_required": validation_required,
                "validation_status": validation_status,
                "validation_summary": validation_summary,
                "next_steps": next_steps,
                "remaining_steps": next_steps,
                "last_execution_result": result,
                "manager_constraints": self._get_manager_constraints(thread_id),
            },
        )

    def _delegation_prefix(self, user_language: str, delegated_result: dict) -> str:
        internal_payload = delegated_result.get("internal_payload") or {}
        task = str(internal_payload.get("task", "")).lower()
        if "review" in task:
            return self.language_policy.user_text(
                user_language,
                "Ich delegiere das an den Kritiker-Agenten.",
                "I am delegating this to the reviewer agent.",
            )
        if "research" in task:
            return self.language_policy.user_text(
                user_language,
                "Ich delegiere das an den Research-Agenten.",
                "I am delegating this to the research agent.",
            )
        return self.language_policy.user_text(
            user_language,
            "Ich delegiere das an den Coding-Agenten.",
            "I am delegating this to the coding agent.",
        )

    def _format_llm_error_reply(
        self,
        user_language: str,
        agent_label: str,
        model: str,
        code: str,
        details: str,
        status_code: int | None = None,
    ) -> str:
        if user_language == "de":
            reply = (
                f"{agent_label}-Fehler:\n"
                f"- Modell: `{model}`\n"
                f"- Fehlercode: `{code}`\n"
                f"- Details: {details}"
            )
            if status_code is not None:
                reply += f"\n- HTTP-Status: `{status_code}`"
            return reply
        reply = (
            f"{agent_label} error:\n"
            f"- Model: `{model}`\n"
            f"- Error code: `{code}`\n"
            f"- Details: {details}"
        )
        if status_code is not None:
            reply += f"\n- HTTP status: `{status_code}`"
        return reply

    def _store_error_response(self, thread_id: str, reply: str, meta: dict) -> dict:
        assistant_message = self.store.add_message(
            thread_id,
            role="assistant",
            content=reply,
            agent="manager",
            meta=meta,
        )
        thread = self.store.get_thread(thread_id)
        return {
            "thread": thread,
            "message": assistant_message,
            "reply": reply,
            "route": meta.get("route", "error"),
            "approval_request": None,
            "messages": self.store.list_messages(thread_id),
        }

    def _finalize_pending_message(
        self,
        thread_id: str,
        pending_message_id: int,
        reply: str,
        meta: dict,
        route: str,
        approval: dict | None,
    ) -> dict:
        log_event(
            logger,
            "pending_finalize_started",
            thread_id=thread_id,
            pending_message_id=pending_message_id,
            route=route,
            reply_chars=len(reply),
            approval_created=bool(approval),
        )
        current = self.store.update_message(pending_message_id)
        log_event(
            logger,
            "pending_finalize_current_loaded",
            thread_id=thread_id,
            pending_message_id=pending_message_id,
            current_found=current is not None,
        )
        current_meta = dict(current.get("meta") or {}) if current else {}
        if current_meta.get("final_decision") == "timeout":
            log_event(
                logger,
                "pending_finalize_skipped_due_to_timeout",
                thread_id=thread_id,
                pending_message_id=pending_message_id,
            )
            thread = self.store.get_thread(thread_id)
            messages = self.store.list_messages(thread_id)
            return {
                "thread": thread,
                "message": current,
                "reply": current.get("content", "") if current else reply,
                "route": current_meta.get("route", "error"),
                "approval_request": None,
                "messages": messages,
            }
        log_event(
            logger,
            "pending_finalize_store_update_started",
            thread_id=thread_id,
            pending_message_id=pending_message_id,
        )
        pending_message = self.store.update_message(
            pending_message_id,
            content=reply,
            agent="manager",
            meta=meta,
        )
        log_event(
            logger,
            "pending_finalize_store_update_completed",
            thread_id=thread_id,
            pending_message_id=pending_message_id,
            message_found=pending_message is not None,
        )
        thread = self.store.get_thread(thread_id)
        log_event(
            logger,
            "pending_finalize_thread_loaded",
            thread_id=thread_id,
            pending_message_id=pending_message_id,
            thread_found=thread is not None,
        )
        messages = self.store.list_messages(thread_id)
        log_event(
            logger,
            "pending_finalize_completed",
            thread_id=thread_id,
            pending_message_id=pending_message_id,
            route=route,
            message_count=len(messages),
        )
        return {
            "thread": thread,
            "message": pending_message,
            "reply": reply,
            "route": route,
            "approval_request": approval,
            "messages": messages,
        }

    def finalize_pending_timeout(
        self,
        thread_id: str,
        pending_message_id: int,
        user_message: str,
        *,
        timeout_seconds: float,
        worker_label: str | None = None,
    ) -> dict:
        language_context = self.language_policy.build_context(user_message)
        label = worker_label or self.language_policy.user_text(
            language_context.user_language,
            "ein interner Agent",
            "an internal agent",
        )
        if language_context.user_language == "de":
            reply = (
                "Die Verarbeitung wurde automatisch abgebrochen, weil ein interner Schritt zu lange "
                "keine verwertbare Antwort geliefert hat.\n"
                f"- Betroffener Schritt: {label}\n"
                f"- Timeout: `{int(timeout_seconds)}` Sekunden\n"
                "- Nächster Schritt: Bitte den Lauf erneut versuchen oder das Debug-Log prüfen."
            )
        else:
            reply = (
                "Processing was stopped automatically because an internal step took too long without "
                "returning a usable result.\n"
                f"- Affected step: {label}\n"
                f"- Timeout: `{int(timeout_seconds)}` seconds\n"
                "- Next step: Retry the run or inspect the debug trace."
            )
        log_event(
            logger,
            "pending_timeout_finalization_started",
            thread_id=thread_id,
            pending_message_id=pending_message_id,
            timeout_seconds=timeout_seconds,
            worker_label=label,
        )
        return self._finalize_pending_message(
            thread_id=thread_id,
            pending_message_id=pending_message_id,
            reply=reply,
            meta={
                "route": "error",
                "approval_request_id": None,
                "planning_note": None,
                "manager_source": "timeout",
                "fallback_reason": None,
                "llm_decision": None,
                "final_decision": "timeout",
                "structured_plan_used": False,
                "fallback_to_heuristic": False,
                "structured_action_count": 0,
                "plan_validation_success": False,
                "processing": False,
                "processing_comment": self.language_policy.user_text(
                    language_context.user_language,
                    "Verarbeitung wegen Timeout abgebrochen",
                    "Processing stopped because of a timeout",
                ),
                "language_policy": {
                    "user_language": language_context.user_language,
                    "internal_language": "en",
                },
                "internal_payload": {
                    "language": "en",
                    "thread_id": thread_id,
                    "task": language_context.internal_message,
                    "source": "timeout",
                    "error": {
                        "code": "chat_processing_timeout",
                        "message": f"Background processing exceeded {timeout_seconds} seconds.",
                        "worker_label": label,
                    },
                },
            },
            route="error",
            approval=None,
        )

    def _update_pending_status(self, pending_message_id: int, comment: str) -> None:
        log_event(
            logger,
            "pending_status_update_started",
            pending_message_id=pending_message_id,
            comment=comment,
        )
        current = self.store.update_message(pending_message_id)
        if current is None:
            log_event(
                logger,
                "pending_status_update_missing_message",
                pending_message_id=pending_message_id,
            )
            return
        meta = dict(current.get("meta") or {})
        meta["processing"] = True
        meta["processing_comment"] = comment
        self.store.update_message(pending_message_id, meta=meta)
        log_event(
            logger,
            "pending_status_update_completed",
            pending_message_id=pending_message_id,
            comment=comment,
        )

    def _processing_comment_for_route(self, decision: ManagerDecision, user_language: str) -> str:
        if decision == ManagerDecision.DIRECT:
            return self.language_policy.user_text(
                user_language,
                "Manager formuliert die Antwort",
                "Manager is composing the reply",
            )
        if decision == ManagerDecision.CODING:
            return self.language_policy.user_text(
                user_language,
                "Coding-Agent wird beauftragt",
                "Delegating to the coding agent",
            )
        if decision == ManagerDecision.PLAN:
            return self.language_policy.user_text(
                user_language,
                "Projektplan wird ausgearbeitet",
                "Project plan is being prepared",
            )
        if decision == ManagerDecision.RESEARCH:
            return self.language_policy.user_text(
                user_language,
                "Research-Agent analysiert die Anfrage",
                "Research agent is analyzing the request",
            )
        if decision == ManagerDecision.REVIEW:
            return self.language_policy.user_text(
                user_language,
                "Kritiker-Agent prüft die Anfrage",
                "Reviewer agent is reviewing the request",
            )
        return self.language_policy.user_text(
            user_language,
            "Verarbeitung läuft",
            "Processing",
        )
