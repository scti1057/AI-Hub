import logging
import re
from uuid import uuid4

from ai_hub.language_policy import LanguagePolicy
from ai_hub.logging_config import log_event, set_request_id, reset_request_id, get_request_id, setup_logging
from ai_hub.memory.history import format_thread_history
from ai_hub.memory.store import HubStore
from ai_hub.llm.manager_planner import ManagerPlanner
from ai_hub.orchestration.delegation import DelegationService
from ai_hub.orchestration.router import ManagerRouter
from ai_hub.schemas.coding_delegation import CodingDelegationPlan
from ai_hub.schemas.manager_plan import ManagerPlan
from ai_hub.state import ApprovalStatus, ManagerDecision
from ai_hub.tools.code_runner import execute_python_approval
from ai_hub.tools.code_runner import ExecutionPolicyError
from ai_hub.tools.file_tools import WorkspaceSecurityError
from ai_hub.tools.push_notify import send_notification


logger = logging.getLogger(__name__)
setup_logging()

ROLE_CONFLICT_PATTERNS = (
    r"\bich bin der coding-agent\b",
    r"\bi am the coding agent\b",
    r"\bich bin der research-agent\b",
    r"\bi am the research agent\b",
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


class ManagerWorkflow:
    def __init__(
        self,
        store: HubStore | None = None,
        planner: ManagerPlanner | None = None,
    ) -> None:
        self.store = store or HubStore()
        self.language_policy = LanguagePolicy()
        self.router = ManagerRouter()
        self.delegation = DelegationService()
        self.planner = planner or ManagerPlanner()

    def handle_chat(self, thread_id: str | None, user_message: str) -> dict:
        token = None
        if get_request_id() == "-":
            token = set_request_id(f"chat-{uuid4().hex[:12]}")
        language_context = self.language_policy.build_context(user_message)
        thread = self.store.ensure_thread(thread_id)
        cleaned_message = language_context.user_message.strip()
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
            history = self.store.list_messages(thread["id"])
            history_text = format_thread_history(history[:-1], limit=6)
            planning_note = self._plan_safely(
                history_text,
                language_context.internal_message,
                language_context.user_language,
            )
            route = self._resolve_route(cleaned_message, planning_note)
            llm_decision = self._llm_decision(planning_note)
            approval = None
            delegated_result = None
            manager_source = "fallback"
            fallback_reason = None
            coding_structured_plan = None
            structured_plan_used = False
            fallback_to_heuristic = False
            structured_action_count = 0
            plan_validation_success = False

            if route.decision == ManagerDecision.DIRECT:
                reply, manager_source, direct_fallback_reason = self._direct_reply(
                    cleaned_message,
                    language_context.user_language,
                    planning_note,
                )
                fallback_reason = direct_fallback_reason or self._fallback_reason(planning_note)
            else:
                if route.decision == ManagerDecision.CODING:
                    coding_structured_plan, structured_plan_used, fallback_to_heuristic, plan_validation_success = (
                        self._resolve_coding_structured_plan(cleaned_message, planning_note)
                    )
                    structured_action_count = (
                        len(coding_structured_plan.actions.actions) if coding_structured_plan is not None else 0
                    )
                delegated_result = self.delegation.execute(
                    route.decision,
                    thread["id"],
                    cleaned_message,
                    history,
                    internal_task=self._worker_task(cleaned_message, planning_note, language_context.internal_message),
                    structured_plan=coding_structured_plan,
                )
                reply = self._manager_wrap(route.reason, delegated_result, planning_note, language_context.user_language)
                manager_source = "ollama" if planning_note and planning_note.get("source") == "ollama" else "fallback"
                fallback_reason = self._fallback_reason(planning_note)
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
                                language_context.user_language,
                                "AI Hub Freigabe erforderlich",
                                "AI Hub approval required",
                            ),
                            body=self.language_policy.user_text(
                                language_context.user_language,
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
                        "user_language": language_context.user_language,
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
        finally:
            if token is not None:
                reset_request_id(token)

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
        if approval["status"] == ApprovalStatus.EXECUTED.value:
            note = (
                f"Ausgeführt: `{approval['command']['preview']}`\n"
                f"Returncode: {result['returncode']}"
            )
        else:
            note = (
                f"Ausführung fehlgeschlagen oder blockiert: `{approval['command']['preview']}`\n"
                f"Grund: {result.get('stderr', 'unbekannt')}"
            )
        self.store.add_message(
            approval["thread_id"],
            role="assistant",
            content=note,
            agent="manager",
            meta={"approval_request_id": approval_id, "execution_result": result},
        )
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
        lowered = user_message.lower()
        if "was kannst du" in lowered or "aktuell in diesem system" in lowered or "fähigkeiten" in lowered:
            return (
                self.language_policy.user_text(
                    user_language,
                    "Aktuell kann ich dir dabei helfen:\n"
                    "- Ich kann Threads verwalten, Chatverläufe speichern und Freigaben koordinieren.\n"
                    "- Ich kann im erlaubten Coding-Workspace Dateien lesen, schreiben und Ordner anlegen.\n"
                    "- Python-Ausführung starte ich nur nach deiner Freigabe.\n"
                    "- Zugriffe auf sensible Pfade oder außerhalb der Sandbox blockiere ich.",
                    "I can currently help with:\n"
                    "- I can manage threads, persist chat history, and coordinate approvals.\n"
                    "- I can read and write files inside the allowed coding workspace.\n"
                    "- I only run Python after your approval.\n"
                    "- I block sensitive paths and anything outside the sandbox.",
                ),
                "fallback",
                "implausible_llm_reply" if plan is not None else self._fallback_reason(planning_note),
            )

        return (
            self.language_policy.user_text(
                user_language,
                f"Verstanden. {user_message}",
                f"Understood. {user_message}",
            ),
            "fallback",
            "implausible_llm_reply" if plan is not None else self._fallback_reason(planning_note),
        )

    def _manager_wrap(
        self,
        reason: str,
        delegated_result: dict,
        planning_note: dict | None,
        user_language: str,
    ) -> str:
        status = delegated_result.get("status")
        reply = delegated_result["reply"]
        if status == "blocked":
            return reply
        manager_prefix = self._delegation_prefix(user_language, delegated_result)
        if status == "completed":
            return f"{manager_prefix}\n{reply}".strip()
        if status == "approval_required":
            return f"{manager_prefix}\n{reply}".strip()
        if status == "completed_with_approval":
            return f"{manager_prefix}\n{reply}".strip()
        return self.language_policy.user_text(
            user_language,
            f"Delegiert: {reason}\n{reply}",
            f"Delegated: {reason}\n{reply}",
        )

    def _plan_safely(self, history_text: str, user_message: str, user_language: str) -> dict | None:
        try:
            return self.planner.plan(
                history_text=history_text,
                user_message=user_message,
                user_language=user_language,
            )
        except Exception as exc:
            logger.warning("manager_planner_fallback error=%s", exc)
            return {
                "enabled": False,
                "error": str(exc),
                "source": "fallback",
            }

    def _resolve_route(self, user_message: str, planning_note: dict | None) -> "RouteDecision":
        fallback_route = self.router.decide(user_message)
        plan = self._llm_plan(planning_note)
        if plan is None:
            return fallback_route
        llm_route = self._coerce_route_decision(plan.decision, plan.reason, fallback_route)
        if llm_route is None:
            return fallback_route
        if self._is_strategic_request(user_message) and not self._has_explicit_workspace_action(user_message):
            return fallback_route
        if fallback_route.decision == ManagerDecision.DIRECT and llm_route.decision != ManagerDecision.DIRECT:
            if not self._has_explicit_workspace_action(user_message):
                return fallback_route
        if fallback_route.decision != ManagerDecision.DIRECT and llm_route.decision == ManagerDecision.DIRECT:
            if self._has_explicit_workspace_action(user_message):
                return fallback_route
        return llm_route

    def _coerce_route_decision(
        self,
        decision_value: str,
        reason: str,
        fallback_route: "RouteDecision",
    ) -> "RouteDecision | None":
        try:
            decision = ManagerDecision(decision_value)
            return type(fallback_route)(decision=decision, reason=reason)
        except Exception:
            return None

    def _llm_decision(self, planning_note: dict | None) -> str | None:
        plan = self._llm_plan(planning_note)
        if plan is None:
            return None
        return plan.decision

    def _has_explicit_workspace_action(self, user_message: str) -> bool:
        lowered = user_message.lower()
        return any(re.search(pattern, lowered) for pattern in EXPLICIT_WORKSPACE_ACTION_PATTERNS)

    def _llm_plan(self, planning_note: dict | None) -> ManagerPlan | None:
        if not planning_note:
            return None
        plan = planning_note.get("plan")
        if isinstance(plan, ManagerPlan):
            return plan
        return None

    def _worker_task(
        self,
        user_message: str,
        planning_note: dict | None,
        fallback_internal_message: str,
    ) -> str:
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
                return plan.coding_plan, True, False, True
            return None, False, True, False
        fallback_plan = self.delegation.prepare_coding_plan(user_message)
        if fallback_plan.actions.actions:
            return fallback_plan, True, False, True
        return None, False, True, False

    def _serialize_planning_note(self, planning_note: dict | None) -> dict | None:
        if planning_note is None:
            return None
        serialized = dict(planning_note)
        plan = serialized.get("plan")
        if isinstance(plan, ManagerPlan):
            serialized["plan"] = plan.model_dump()
        return serialized

    def _fallback_reason(self, planning_note: dict | None) -> str | None:
        if not planning_note:
            return "planner_returned_none"
        if planning_note.get("source") == "ollama":
            return None
        return planning_note.get("error") or "planner_not_used"

    def _is_plausible_manager_reply(self, reply: str) -> bool:
        lowered = reply.lower().strip()
        if not lowered:
            return False
        return not any(re.search(pattern, lowered) for pattern in ROLE_CONFLICT_PATTERNS)

    def _is_strategic_request(self, user_message: str) -> bool:
        lowered = user_message.lower()
        return any(pattern in lowered for pattern in STRATEGIC_USER_PATTERNS)

    def _delegation_prefix(self, user_language: str, delegated_result: dict) -> str:
        internal_payload = delegated_result.get("internal_payload") or {}
        task = str(internal_payload.get("task", "")).lower()
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
