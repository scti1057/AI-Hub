from ai_hub.orchestration.workflow import ManagerWorkflow


class ManagerAgent:
    def __init__(self) -> None:
        self.workflow = ManagerWorkflow()

    def run(self, user_task: str, thread_id: str | None = None) -> dict:
        return self.workflow.handle_chat(thread_id=thread_id, user_message=user_task)
