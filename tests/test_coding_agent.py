import importlib

import pytest


def configure_workspace(monkeypatch, tmp_path):
    workspace_root = tmp_path / "workspaces"

    import ai_hub.config as config
    import ai_hub.tools.file_tools as file_tools

    monkeypatch.setattr(config, "CODING_WORKSPACE_ROOT", workspace_root)
    monkeypatch.setattr(file_tools, "CODING_WORKSPACE_ROOT", workspace_root)

    importlib.reload(file_tools)
    return file_tools


class FakeCodingClient:
    def __init__(self, response: str, fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.calls: list[dict] = []

    def generate(self, model: str, prompt: str, temperature: float = 0.2) -> str:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
            }
        )
        if self.fail:
            raise RuntimeError("ollama unavailable")
        return self.response


def test_workspace_blocks_sensitive_and_parent_paths(monkeypatch, tmp_path):
    file_tools = configure_workspace(monkeypatch, tmp_path)

    with pytest.raises(file_tools.WorkspaceSecurityError):
        file_tools.resolve_workspace_path("thread-1", "../outside.py")

    with pytest.raises(file_tools.WorkspaceSecurityError):
        file_tools.resolve_workspace_path("thread-1", "secrets/token.txt")


def test_workspace_allows_safe_read_write(monkeypatch, tmp_path):
    file_tools = configure_workspace(monkeypatch, tmp_path)

    write_result = file_tools.write_file("thread-2", "src/main.py", "print('ok')\n")
    content = file_tools.read_file("thread-2", "src/main.py")
    entries = file_tools.list_files("thread-2", "src")

    assert write_result["path"] == "src/main.py"
    assert "print('ok')" in content
    assert entries[0]["name"] == "main.py"


def test_coding_agent_reports_blocked_dotenv_access(monkeypatch, tmp_path):
    configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent

    agent = CodingAgent()
    result = agent.handle_task("thread-3", "Bitte lies .env", [])

    assert result["status"] == "blocked"
    assert ".env" in result["reply"]


def test_coding_agent_can_build_action_batch_from_llm(monkeypatch, tmp_path):
    configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent

    client = FakeCodingClient(
        """
        {
          "actions": [
            {"action_type": "make_directory", "path": "demo"},
            {"action_type": "create_file", "path": "demo/hello.py", "content": "print('hi')\\n", "content_inferred": false},
            {"action_type": "request_execution", "target": "demo/hello.py"}
          ]
        }
        """
    )

    agent = CodingAgent(client=client)
    batch = agent.build_action_batch_from_llm("Bitte erstelle demo/hello.py und beantrage die Ausführung.")

    assert client.calls[0]["model"] == "qwen3-coder:30b"
    assert [action.action_type for action in batch.actions] == [
        "make_directory",
        "create_file",
        "request_execution",
    ]


def test_coding_agent_falls_back_to_heuristics_when_llm_fails(monkeypatch, tmp_path):
    configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent

    client = FakeCodingClient("{}", fail=True)
    agent = CodingAgent(client=client)
    result = agent.handle_task(
        thread_id="thread-llm-fallback",
        user_task="Erstelle einen Unterordner demo und darin notes.txt",
        history=[],
    )

    assert result["status"] == "error"
    assert "Coding-Agent Fehler:" in result["reply"]
    assert result["internal_payload"]["source"] == "error"


def test_coding_agent_executes_llm_planned_actions(monkeypatch, tmp_path):
    configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent
    from ai_hub.tools.file_tools import read_file

    client = FakeCodingClient(
        """
        {
          "actions": [
            {"action_type": "make_directory", "path": "demo"},
            {"action_type": "create_file", "path": "demo/from_llm.txt", "content": "OK", "content_inferred": false}
          ]
        }
        """
    )

    agent = CodingAgent(client=client)
    result = agent.handle_task(
        thread_id="thread-llm-plan",
        user_task="Bitte lege eine Datei demo/from_llm.txt an.",
        history=[],
    )

    assert result["status"] == "completed"
    assert [item["action_type"] for item in result["actions_executed"]] == ["make_directory", "create_file"]
    assert read_file("thread-llm-plan", "demo/from_llm.txt") == "OK"
    assert "Self-check:" in result["internal_summary"]
    assert result["internal_payload"]["self_check"]["inspected_files"][0]["path"] == "demo/from_llm.txt"


def test_coding_agent_self_check_flags_missing_runtime_validation(monkeypatch, tmp_path):
    configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent

    client = FakeCodingClient(
        """
        {
          "actions": [
            {"action_type": "create_file", "path": "app/main.py", "content": "print('ok')\\n", "content_inferred": false}
          ]
        }
        """
    )

    agent = CodingAgent(client=client)
    result = agent.handle_task(
        thread_id="thread-self-check",
        user_task="Bitte erstelle app/main.py",
        history=[],
    )

    self_check = result["internal_payload"]["self_check"]
    assert result["status"] == "completed"
    assert self_check["inspected_files"][0]["path"] == "app/main.py"
    assert any("runtime validation" in item.lower() for item in self_check["follow_up"])
