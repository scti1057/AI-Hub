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


def test_coding_agent_uses_internal_task_for_llm_planning(monkeypatch, tmp_path):
    configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent

    client = FakeCodingClient(
        """
        {
          "actions": [
            {"action_type": "create_file", "path": "src/main.py", "content": "print('from step')\\n", "content_inferred": false}
          ]
        }
        """
    )

    agent = CodingAgent(client=client)
    result = agent.handle_task(
        thread_id="thread-internal-task",
        user_task="Please build the whole project.",
        history=[],
        internal_task="Current step: create src/main.py and nothing else.",
    )

    assert result["status"] == "completed"
    assert "Current step: create src/main.py and nothing else." in client.calls[0]["prompt"]
    assert "Please build the whole project." in client.calls[0]["prompt"]


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


def test_coding_agent_rejects_placeholder_only_empty_python_structured_plan(monkeypatch, tmp_path):
    configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent
    from ai_hub.schemas.coding_actions import CodingActionBatch, CreateFileAction

    agent = CodingAgent()
    result = agent.handle_task(
        thread_id="thread-empty-python-plan",
        user_task="Please implement src/main.py.",
        history=[],
        structured_plan=CodingActionBatch(
            actions=[CreateFileAction(path="src/main.py", content="", content_inferred=True)]
        ),
    )

    assert result["status"] == "error"
    assert "placeholder-only empty file writes" in result["reply"] or "placeholder-only empty file writes" in result["internal_payload"]["error"]["message"]


def test_coding_agent_keeps_explicit_missing_read_blocked(monkeypatch, tmp_path):
    configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent
    from ai_hub.schemas.coding_actions import CodingActionBatch, ReadFileAction

    agent = CodingAgent()
    result = agent.handle_task(
        thread_id="thread-explicit-missing-read",
        user_task="Please inspect src/main.py",
        history=[],
        structured_plan=CodingActionBatch(actions=[ReadFileAction(path="src/main.py")]),
    )

    assert result["status"] == "blocked"
    assert "Not found" in result["reply"] or "Nicht gefunden" in result["reply"]


def test_coding_agent_skips_missing_integration_read_inside_mutating_batch(monkeypatch, tmp_path):
    file_tools = configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent
    from ai_hub.schemas.coding_actions import CodingActionBatch, CreateFileAction, ReadFileAction

    file_tools.write_file("thread-skip-read", "src/game_engine.py", "class Engine:\n    pass\n")
    agent = CodingAgent()
    result = agent.handle_task(
        thread_id="thread-skip-read",
        user_task="Implement the next bounded UI step.",
        history=[],
        structured_plan=CodingActionBatch(
            actions=[
                CreateFileAction(path="src/gui_components.py", content="print('ui')\n", content_inferred=False),
                ReadFileAction(path="src/main.py"),
                ReadFileAction(path="src/game_engine.py"),
            ]
        ),
    )

    assert result["status"] == "completed"
    assert any(item.get("target") == "src/main.py" and item["action_type"] == "read_file" for item in result["actions_executed"])
    skipped_entry = next(item for item in result["actions_executed"] if item.get("target") == "src/main.py")
    assert skipped_entry["details"]["skipped"] is True
    assert "Skipped read" in result["reply"] or "Lesen übersprungen" in result["reply"]


def test_coding_agent_injects_contract_reads_before_mutation(monkeypatch, tmp_path):
    file_tools = configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent
    from ai_hub.schemas.coding_actions import CodingActionBatch, CreateFileAction

    file_tools.write_file("thread-contract-reads", "src/main.py", "print('old')\n")
    file_tools.write_file("thread-contract-reads", "tests/test_main.py", "def test_ok():\n    assert True\n")

    agent = CodingAgent()
    result = agent.handle_task(
        thread_id="thread-contract-reads",
        user_task="Implement the next bounded step.",
        history=[],
        internal_task=(
            "Update src/app.py.\n"
            "Relevant existing paths: src/main.py\n"
            "Relevant validation or entry-point paths: tests/test_main.py"
        ),
        structured_plan=CodingActionBatch(
            actions=[CreateFileAction(path="src/app.py", content="print('new')\n", content_inferred=False)]
        ),
    )

    assert result["status"] == "completed"
    executed_types = [(item["action_type"], item.get("target")) for item in result["actions_executed"]]
    assert executed_types[:2] == [
        ("read_file", "src/main.py"),
        ("read_file", "tests/test_main.py"),
    ]


def test_coding_agent_self_check_reports_contract_alignment(monkeypatch, tmp_path):
    file_tools = configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent
    from ai_hub.schemas.coding_actions import CodingActionBatch, CreateFileAction

    file_tools.write_file("thread-contract-self-check", "src/main.py", "print('old')\n")
    file_tools.write_file("thread-contract-self-check", "tests/test_main.py", "def test_ok():\n    assert True\n")

    agent = CodingAgent()
    result = agent.handle_task(
        thread_id="thread-contract-self-check",
        user_task="Implement the next bounded step.",
        history=[],
        internal_task=(
            "Update src/app.py.\n"
            "Relevant existing paths: src/main.py\n"
            "Relevant validation or entry-point paths: tests/test_main.py"
        ),
        structured_plan=CodingActionBatch(
            actions=[CreateFileAction(path="src/app.py", content="print('new')\n", content_inferred=False)]
        ),
    )

    self_check = result["internal_payload"]["self_check"]
    assert self_check["contract_paths_inspected"] == ["src/main.py"]
    assert self_check["validation_paths_inspected"] == ["tests/test_main.py"]
    assert "Contract-aligned reads: src/main.py." in self_check["summary"]
    assert "Validation-context reads: tests/test_main.py." in self_check["summary"]


def test_coding_agent_normalizes_pytest_command_string_request(monkeypatch, tmp_path):
    file_tools = configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent
    from ai_hub.schemas.coding_actions import CodingActionBatch, RequestExecutionAction

    file_tools.write_file("thread-pytest-command", "tests/test_game_engine.py", "def test_ok():\n    assert True\n")
    agent = CodingAgent()
    result = agent.handle_task(
        thread_id="thread-pytest-command",
        user_task="Run the test file.",
        history=[],
        structured_plan=CodingActionBatch(
            actions=[RequestExecutionAction(target="python -m pytest tests/test_game_engine.py -v")]
        ),
    )

    assert result["status"] == "approval_required"
    approval = result["approval_request"]
    assert approval is not None
    assert approval["command"]["argv"] == ["-m", "pytest", "tests/test_game_engine.py", "-v"]
    assert approval["command"]["preview"] == "python -m pytest tests/test_game_engine.py -v"


def test_coding_agent_normalizes_workspace_test_file_to_pytest(monkeypatch, tmp_path):
    file_tools = configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent
    from ai_hub.schemas.coding_actions import CodingActionBatch, RequestExecutionAction

    file_tools.write_file("thread-test-file-run", "tests/test_game_engine.py", "def test_ok():\n    assert True\n")
    agent = CodingAgent()
    result = agent.handle_task(
        thread_id="thread-test-file-run",
        user_task="Run the unit test file.",
        history=[],
        structured_plan=CodingActionBatch(
            actions=[RequestExecutionAction(target="tests/test_game_engine.py")]
        ),
    )

    assert result["status"] == "approval_required"
    approval = result["approval_request"]
    assert approval is not None
    assert approval["command"]["argv"] == ["-m", "pytest", "tests/test_game_engine.py", "-v"]
    assert approval["command"]["preview"] == "python -m pytest tests/test_game_engine.py -v"


def test_dependency_analysis_ignores_stdlib_and_local_modules(monkeypatch, tmp_path):
    file_tools = configure_workspace(monkeypatch, tmp_path)

    from ai_hub.tools.dependency_tools import analyze_workspace_dependencies

    file_tools.write_file(
        "thread-dependency-audit",
        "src/main.py",
        "import tkinter\nimport requests\nfrom app import helper\n",
    )
    file_tools.write_file("thread-dependency-audit", "src/app/__init__.py", "")
    file_tools.write_file("thread-dependency-audit", "src/app/helper.py", "VALUE = 1\n")

    result = analyze_workspace_dependencies("thread-dependency-audit")

    assert "requests" in result["missing_requirements"]
    assert "tkinter" not in result["missing_requirements"]
    assert "app" not in result["missing_requirements"]


def test_dependency_analysis_ignores_src_layout_package_root(monkeypatch, tmp_path):
    file_tools = configure_workspace(monkeypatch, tmp_path)

    from ai_hub.tools.dependency_tools import analyze_workspace_dependencies

    file_tools.write_file(
        "thread-src-layout",
        "src/main.py",
        "from src.constants import VALUE\n",
    )
    file_tools.write_file("thread-src-layout", "src/constants.py", "VALUE = 1\n")

    result = analyze_workspace_dependencies("thread-src-layout")

    assert "src" not in result["missing_requirements"]
    assert result["packages_to_install"] == []
