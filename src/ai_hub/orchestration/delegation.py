from ai_hub.language_policy import LanguagePolicy
from ai_hub.agents.coding_agent import CodingAgent
from ai_hub.agents.research_agent import ResearchAgent
from ai_hub.schemas.coding_delegation import CodingDelegationPlan
from ai_hub.state import ManagerDecision


class DelegationService:
    def __init__(self) -> None:
        self.language_policy = LanguagePolicy()
        self.coding_agent = CodingAgent(language_policy=self.language_policy)
        self.research_agent = ResearchAgent()

    def execute(
        self,
        decision: ManagerDecision,
        thread_id: str,
        user_task: str,
        history: list[dict],
        internal_task: str | None = None,
        structured_plan: CodingDelegationPlan | None = None,
    ) -> dict:
        if decision == ManagerDecision.CODING:
            return self.coding_agent.handle_task(
                thread_id,
                user_task,
                history,
                internal_task=internal_task,
                structured_plan=structured_plan,
            )
        if decision == ManagerDecision.RESEARCH:
            return self.research_agent.handle_task(thread_id, user_task, history, internal_task=internal_task)
        raise ValueError(f"Unsupported delegation decision: {decision}")

    def prepare_coding_plan(self, user_task: str) -> CodingDelegationPlan:
        batch = self.coding_agent.build_action_batch_from_text(user_task)
        return CodingDelegationPlan(
            summary="Fallback structured coding delegation plan.",
            rationale="Derived from the user request with conservative local parsing.",
            approval_needed=any(action.action_type == "request_execution" for action in batch.actions),
            actions=batch,
        )
