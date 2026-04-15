import importlib


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
    assert client.calls[0]["model"] == "deepseek-r1:32b-qwen-distill-q4_K_M"
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
