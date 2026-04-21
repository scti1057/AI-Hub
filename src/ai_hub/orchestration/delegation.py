import json
import logging

from ai_hub.language_policy import LanguagePolicy
from ai_hub.agents.coding_agent import CodingAgent
from ai_hub.agents.explorer_agent import ExplorerAgent
from ai_hub.agents.research_agent import ResearchAgent
from ai_hub.agents.reviewer_agent import ReviewerAgent
from ai_hub.logging_config import log_event, log_text_block, setup_logging
from ai_hub.memory.store import HubStore
from ai_hub.schemas.coding_delegation import CodingDelegationPlan
from ai_hub.state import ManagerDecision


logger = logging.getLogger(__name__)
setup_logging()


class DelegationService:
    def __init__(self, store: HubStore | None = None) -> None:
        self.language_policy = LanguagePolicy()
        self.coding_agent = CodingAgent(language_policy=self.language_policy)
        self.explorer_agent = ExplorerAgent(language_policy=self.language_policy, store=store)
        self.research_agent = ResearchAgent(store=store)
        self.reviewer_agent = ReviewerAgent()

    def execute(
        self,
        decision: ManagerDecision,
        thread_id: str,
        user_task: str,
        history: list[dict],
        internal_task: str | None = None,
        structured_plan: CodingDelegationPlan | None = None,
    ) -> dict:
        log_event(
            logger,
            "delegation_execute_started",
            thread_id=thread_id,
            decision=decision.value,
            internal_task_chars=len(internal_task or ""),
            history_items=len(history),
            structured_plan=bool(structured_plan),
        )
        if internal_task:
            log_text_block(
                logger,
                "delegation_internal_task",
                internal_task,
                thread_id=thread_id,
                decision=decision.value,
            )
        if decision == ManagerDecision.CODING:
            result = self.coding_agent.handle_task(
                thread_id,
                user_task,
                history,
                internal_task=internal_task,
                structured_plan=structured_plan,
            )
            log_event(
                logger,
                "delegation_execute_completed",
                thread_id=thread_id,
                decision=decision.value,
                status=result.get("status"),
            )
            self._log_result(thread_id, decision, result)
            return result
        if decision == ManagerDecision.RESEARCH:
            result = self.research_agent.handle_task(thread_id, user_task, history, internal_task=internal_task)
            log_event(
                logger,
                "delegation_execute_completed",
                thread_id=thread_id,
                decision=decision.value,
                status=result.get("status"),
            )
            self._log_result(thread_id, decision, result)
            return result
        if decision == ManagerDecision.EXPLORER:
            result = self.explorer_agent.handle_task(thread_id, user_task, history, internal_task=internal_task)
            log_event(
                logger,
                "delegation_execute_completed",
                thread_id=thread_id,
                decision=decision.value,
                status=result.get("status"),
            )
            self._log_result(thread_id, decision, result)
            return result
        if decision == ManagerDecision.REVIEW:
            result = self.reviewer_agent.handle_task(thread_id, user_task, history, internal_task=internal_task)
            log_event(
                logger,
                "delegation_execute_completed",
                thread_id=thread_id,
                decision=decision.value,
                status=result.get("status"),
            )
            self._log_result(thread_id, decision, result)
            return result
        raise ValueError(f"Unsupported delegation decision: {decision}")

    def prepare_coding_plan(self, user_task: str) -> CodingDelegationPlan:
        batch = self.coding_agent.build_action_batch_from_llm(user_task)
        return CodingDelegationPlan(
            summary="Structured coding delegation plan.",
            rationale="Derived from the coding agent model.",
            approval_needed=any(action.action_type == "request_execution" for action in batch.actions),
            actions=batch,
        )

    def _log_result(self, thread_id: str, decision: ManagerDecision, result: dict) -> None:
        compact_payload = {
            "status": result.get("status"),
            "reply": result.get("reply"),
            "user_reply": result.get("user_reply"),
            "internal_summary": result.get("internal_summary"),
            "approval_request_created": result.get("approval_request_created"),
            "approval_request": result.get("approval_request"),
            "actions_executed": result.get("actions_executed"),
            "actions_blocked": result.get("actions_blocked"),
            "tool_results": result.get("tool_results"),
            "internal_payload": result.get("internal_payload"),
        }
        log_text_block(
            logger,
            "delegation_result",
            json.dumps(compact_payload, ensure_ascii=False, indent=2),
            thread_id=thread_id,
            decision=decision.value,
        )
