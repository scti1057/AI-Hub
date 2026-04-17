import logging

from ai_hub.language_policy import LanguagePolicy
from ai_hub.agents.coding_agent import CodingAgent
from ai_hub.agents.research_agent import ResearchAgent
from ai_hub.agents.reviewer_agent import ReviewerAgent
from ai_hub.logging_config import log_event, setup_logging
from ai_hub.schemas.coding_delegation import CodingDelegationPlan
from ai_hub.state import ManagerDecision


logger = logging.getLogger(__name__)
setup_logging()


class DelegationService:
    def __init__(self) -> None:
        self.language_policy = LanguagePolicy()
        self.coding_agent = CodingAgent(language_policy=self.language_policy)
        self.research_agent = ResearchAgent()
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
