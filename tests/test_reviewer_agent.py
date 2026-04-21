import importlib

from ai_hub.config import REVIEWER_MODEL


def configure_paths(monkeypatch, tmp_path):
    db_path = tmp_path / "ai_hub.db"
    workspace_root = tmp_path / "workspaces"

    import ai_hub.config as config
    import ai_hub.memory.sqlite_db as sqlite_db
    import ai_hub.tools.file_tools as file_tools
    import ai_hub.tools.code_runner as code_runner

    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(config, "CODING_WORKSPACE_ROOT", workspace_root)
    monkeypatch.setattr(sqlite_db, "DB_PATH", db_path)
    monkeypatch.setattr(file_tools, "CODING_WORKSPACE_ROOT", workspace_root)

    importlib.reload(sqlite_db)
    importlib.reload(file_tools)
    importlib.reload(code_runner)


class FakeReviewerClient:
    def __init__(self, response: str, fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.calls: list[dict] = []

    def generate(self, model: str, prompt: str, temperature: float = 0.2) -> str:
        self.calls.append({"model": model, "prompt": prompt, "temperature": temperature})
        if self.fail:
            raise RuntimeError("ollama unavailable")
        return self.response


def test_reviewer_agent_uses_reviewer_model(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.reviewer_agent import ReviewerAgent

    client = FakeReviewerClient(
        """
        {
          "summary": "The plan needs a stronger verification step.",
          "findings": ["No explicit test step is defined."],
          "assumptions": ["The implementation may change multiple files."],
          "open_questions": ["Which behavior is most critical to protect?"],
          "recommendation": "Add one focused verification step before execution.",
          "user_reply": "Ich habe den Plan geprüft und den größten Risikopunkt markiert."
        }
        """
    )
    agent = ReviewerAgent(client=client)

    result = agent.handle_task(
        thread_id="thread-v1",
        user_task="Bitte prüfe den Plan kritisch.",
        history=[],
    )

    assert result["status"] == "completed"
    assert client.calls[0]["model"] == REVIEWER_MODEL
    assert "Kritische Punkte:" in result["reply"]
    assert result["internal_payload"]["task"].startswith("review:")


def test_reviewer_agent_falls_back_cleanly(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.reviewer_agent import ReviewerAgent

    client = FakeReviewerClient("{}", fail=True)
    agent = ReviewerAgent(client=client)

    result = agent.handle_task(
        thread_id="thread-v2",
        user_task="Bitte gib mir eine kritische Zweitmeinung.",
        history=[],
    )

    assert result["status"] == "error"
    assert "Kritiker-Agent Fehler:" in result["reply"]
    assert result["internal_payload"]["source"] == "error"


def test_reviewer_agent_parses_step_and_project_completion_separately(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.reviewer_agent import ReviewerAgent

    client = FakeReviewerClient(
        """
        {
          "summary": "The bounded step is implemented, but the broader project still needs testing.",
          "verdict": "done",
          "definition_of_done_met": true,
          "project_status": "ready_to_test",
          "findings": ["No blocking flaw remains in this step."],
          "assumptions": ["The manager only asked for one bounded implementation step."],
          "open_questions": ["Should the next action be an integration test run?"],
          "repair_tasks": [],
          "project_completion_notes": ["The step is done, but the overall project should still be tested end to end."],
          "recommendation": "Run the intended validation before calling the whole project complete.",
          "user_reply": "Ich habe Schritt und Gesamtstatus getrennt bewertet."
        }
        """
    )
    agent = ReviewerAgent(client=client)

    result = agent.handle_task(
        thread_id="thread-review-project-status",
        user_task="Bitte prüfe kritisch, ob der Schritt wirklich fertig ist.",
        history=[],
    )

    assert result["status"] == "completed"
    payload = result["internal_payload"]
    assert payload["verdict"] == "done"
    assert payload["definition_of_done_met"] is True
    assert payload["project_status"] == "ready_to_test"
    assert payload["project_completion_notes"] == [
        "The step is done, but the overall project should still be tested end to end."
    ]
    assert "Projektstatus:" in result["reply"]


def test_reviewer_agent_normalizes_validation_aware_project_status(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.reviewer_agent import ReviewerAgent

    client = FakeReviewerClient(
        """
        {
          "summary": "The bounded step is done, but validation still needs to run.",
          "verdict": "done",
          "definition_of_done_met": true,
          "project_status": "ready_for_validation",
          "findings": ["No blocking flaw remains in the bounded step."],
          "assumptions": ["The manager still expects one smoke test before broader rollout."],
          "open_questions": ["Should the next action be the agreed smoke test?"],
          "repair_tasks": [],
          "project_completion_notes": ["The project is ready for validation, but not yet validated ready to test."],
          "recommendation": "Run the agreed validation step next.",
          "user_reply": "Ich habe den Schritt als fertig bewertet, aber die Validierung fehlt noch."
        }
        """
    )
    agent = ReviewerAgent(client=client)

    result = agent.handle_task(
        thread_id="thread-review-validation-status",
        user_task="Bitte prüfe kritisch, ob wir schon testbereit sind.",
        history=[],
    )

    assert result["status"] == "completed"
    payload = result["internal_payload"]
    assert payload["project_status"] == "ready_for_validation"
    assert payload["definition_of_done_met"] is True
    assert "Projektstatus:" in result["reply"]
