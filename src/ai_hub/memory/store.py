from ai_hub.memory import sqlite_db
from ai_hub.state import ApprovalStatus


class HubStore:
    def __init__(self) -> None:
        sqlite_db.init_db()

    def ensure_thread(self, thread_id: str | None, title: str | None = None) -> dict:
        if thread_id:
            thread = sqlite_db.get_thread(thread_id)
            if thread:
                return thread
        return sqlite_db.create_thread(title or "Neuer Chat")

    def list_threads(self, include_archived: bool = True) -> list[dict]:
        return sqlite_db.list_threads(include_archived=include_archived)

    def get_thread(self, thread_id: str) -> dict | None:
        return sqlite_db.get_thread(thread_id)

    def rename_thread_from_first_message(self, thread_id: str, first_message: str) -> dict | None:
        thread = sqlite_db.get_thread(thread_id)
        if thread is None:
            return None
        if thread["title"] != "Neuer Chat":
            return thread
        title = " ".join(first_message.strip().split())[:80] or "Neuer Chat"
        return sqlite_db.update_thread_title(thread_id, title)

    def archive_thread(self, thread_id: str) -> dict | None:
        return sqlite_db.archive_thread(thread_id)

    def delete_thread(self, thread_id: str) -> bool:
        return sqlite_db.delete_thread(thread_id)

    def reset_thread(self, thread_id: str) -> dict | None:
        return sqlite_db.reset_thread(thread_id)

    def add_message(self, thread_id: str, role: str, content: str, agent: str | None = None, meta: dict | None = None) -> dict:
        return sqlite_db.add_message(thread_id, role, content, agent=agent, meta=meta)

    def update_message(
        self,
        message_id: int,
        *,
        content: str | None = None,
        agent: str | None = None,
        meta: dict | None = None,
    ) -> dict | None:
        return sqlite_db.update_message(
            message_id,
            content=content,
            agent=agent,
            meta=meta,
        )

    def list_messages(self, thread_id: str) -> list[dict]:
        return sqlite_db.list_messages(thread_id)

    def get_artifact(self, thread_id: str, kind: str) -> dict | None:
        return sqlite_db.get_thread_artifact(thread_id, kind)

    def list_artifacts(self, thread_id: str) -> list[dict]:
        return sqlite_db.list_thread_artifacts(thread_id)

    def upsert_artifact(
        self,
        thread_id: str,
        kind: str,
        title: str,
        summary: str,
        content: dict | None = None,
    ) -> dict:
        return sqlite_db.upsert_thread_artifact(
            thread_id=thread_id,
            kind=kind,
            title=title,
            summary=summary,
            content=content,
        )

    def create_approval_request(
        self,
        thread_id: str,
        agent_role: str,
        tool_name: str,
        command: dict,
        rationale: str,
    ) -> dict:
        return sqlite_db.create_approval_request(
            thread_id=thread_id,
            agent_role=agent_role,
            tool_name=tool_name,
            command=command,
            rationale=rationale,
        )

    def get_approval_request(self, approval_id: str) -> dict | None:
        return sqlite_db.get_approval_request(approval_id)

    def list_approvals(self, status: str | None = None, thread_id: str | None = None) -> list[dict]:
        return sqlite_db.list_approval_requests(status=status, thread_id=thread_id)

    def update_approval(self, approval_id: str, status: ApprovalStatus, decision_note: str | None = None, result: dict | None = None) -> dict | None:
        return sqlite_db.update_approval_request(
            approval_id=approval_id,
            status=status,
            decision_note=decision_note,
            result=result,
        )

    def system_status(self) -> dict:
        return {
            "threads_total": sqlite_db.count_threads(),
            "push_subscriptions": sqlite_db.count_push_subscriptions(),
            "pending_approvals": sqlite_db.count_pending_approvals(),
        }
