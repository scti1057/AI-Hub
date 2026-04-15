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
