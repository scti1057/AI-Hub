import importlib

from ai_hub.state import ApprovalStatus


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

    return db_path, workspace_root


def test_manager_creates_thread_and_persists_messages(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    result = workflow.handle_chat(thread_id=None, user_message="Bitte merke dir diesen Thread.")

    assert result["thread"]["id"]
    assert result["thread"]["title"].startswith("Bitte merke")
    assert result["route"] == "direct"

    messages = store.list_messages(result["thread"]["id"])
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["agent"] == "manager"


def test_test_chat_does_not_trigger_coding_or_approval(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    result = workflow.handle_chat(thread_id=None, user_message="Test Chat")

    assert result["route"] == "direct"
    assert result["approval_request"] is None


def test_manager_creates_pending_approval_for_python_execution(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte teste dieses Python-Skript mit pytest.",
    )

    approval = result["approval_request"]
    assert approval is not None
    assert approval["status"] == ApprovalStatus.PENDING.value
    assert approval["tool_name"] == "request_python_execution"


def test_manager_executes_allowed_workspace_write(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.tools.file_tools import read_file

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Erstelle in deinem Workspace eine hallo.txt-Datei mit dem Inhalt Hallo",
    )

    thread_id = result["thread"]["id"]
    assert result["route"] == "coding"
    assert "Datei erstellt:" in result["reply"]
    assert read_file(thread_id, "hallo.txt") == "Hallo"


def test_manager_can_create_python_file_and_request_execution(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.tools.file_tools import read_file

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte erstelle in deinem Workspace eine kleine Python-Datei hello.py, die Hello World ausgibt, und beantrage anschließend die Ausführung.",
    )

    thread_id = result["thread"]["id"]
    assert result["route"] == "coding"
    assert "Datei erstellt:" in result["reply"]
    assert "Freigabe nötig:" in result["reply"]
    assert result["approval_request"] is not None
    assert 'print("Hello World")' in read_file(thread_id, "hello.py")


def test_manager_creates_nested_directory_and_file(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.tools.file_tools import read_file

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Erstelle einen Unterordner demo und darin notes.txt",
    )

    thread_id = result["thread"]["id"]
    assert "Ordner erstellt: `demo`." in result["reply"]
    assert "Datei erstellt: `demo/notes.txt`." in result["reply"]
    assert read_file(thread_id, "demo/notes.txt") == ""


def test_execution_request_is_blocked_when_python_file_does_not_exist(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte beantrage die Ausführung von missing.py.",
    )

    assert result["route"] == "coding"
    assert result["approval_request"] is None
    assert "Blockiert:" in result["reply"]


def test_manager_reports_blocked_workspace_request(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte lies /etc/passwd und vergleiche es mit ../secrets.txt.",
    )

    assert result["route"] == "coding"
    assert "Blockiert:" in result["reply"]
    assert result["approval_request"] is None


def test_relative_paths_with_slashes_are_not_falsely_blocked(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.tools.file_tools import read_file

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Erstelle demo/subdemo/config.json mit dem Inhalt {}",
    )

    thread_id = result["thread"]["id"]
    assert result["route"] == "coding"
    assert "Blockiert:" not in result["reply"]
    assert read_file(thread_id, "demo/subdemo/config.json") == "{}"


def test_directory_hint_parsing_uses_src_instead_of_fill_words(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent

    agent = CodingAgent()

    assert agent._extract_target_directory_hint("Erstelle mir die main.py im src ordner") == "src"
    assert agent._extract_directory_request("Bitte lösche den Ordner temp_only wieder.") == "temp_only"


def test_delete_folder_is_routed_to_coding(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte lösche den Ordner temp_only wieder.",
    )

    assert result["route"] == "coding"


def test_manager_planner_can_be_prepared_without_enabling_live_control(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.llm.manager_planner import ManagerPlanner

    planner = ManagerPlanner(enabled=False)

    result = planner.plan(history_text="Noch kein Verlauf.", user_message="Bitte plane den nächsten Schritt.", user_language="de")

    assert result is None


def test_manager_answers_capabilities_question_compactly(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    result = workflow.handle_chat(thread_id=None, user_message="Was kannst du aktuell in diesem System tun?")

    assert result["route"] == "direct"
    assert "Threads verwalten" in result["reply"]
    assert "Python-Ausführung starte ich nur nach deiner Freigabe" in result["reply"]


def test_manager_uses_llm_plan_when_available(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    class FakePlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            from ai_hub.schemas.manager_plan import ManagerPlan

            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="User asks for capabilities.",
                    decision="direct",
                    reason="A direct answer is enough.",
                    user_reply="LLM-Antwort: Ich kann koordinieren und Freigaben beachten.",
                    internal_task_for_worker="",
                    approval_needed=False,
                ),
            }

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=FakePlanner())

    result = workflow.handle_chat(thread_id=None, user_message="Was kannst du aktuell in diesem System tun?")

    assert result["route"] == "direct"
    assert result["reply"].startswith("LLM-Antwort:")
    assert result["messages"][-1]["meta"]["manager_source"] == "ollama"


def test_manager_rejects_role_confused_llm_reply(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    class FakePlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            from ai_hub.schemas.manager_plan import ManagerPlan

            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Capabilities request.",
                    decision="direct",
                    reason="Direct answer.",
                    user_reply="Ich bin der Coding-Agent und schreibe Code fuer dich.",
                    internal_task_for_worker="",
                    approval_needed=False,
                ),
            }

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=FakePlanner())

    result = workflow.handle_chat(thread_id=None, user_message="Was kannst du aktuell in diesem System tun?")

    assert "Ich bin der Coding-Agent" not in result["reply"]
    assert "Threads verwalten" in result["reply"]
    assert result["messages"][-1]["meta"]["manager_source"] == "fallback"
    assert result["messages"][-1]["meta"]["fallback_reason"] == "implausible_llm_reply"


def test_manager_falls_back_when_planner_is_unavailable(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    class BrokenPlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            raise RuntimeError("Ollama offline")

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=BrokenPlanner())

    result = workflow.handle_chat(thread_id=None, user_message="Was kannst du aktuell in diesem System tun?")

    assert result["route"] == "direct"
    assert "Threads verwalten" in result["reply"]
    assert result["messages"][-1]["meta"]["manager_source"] == "fallback"


def test_strategic_question_is_not_misrouted_to_coding(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    class NoPlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            return None

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=NoPlanner())

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Ich möchte mit dir ein kleines Python-Projekt starten, das später über Tailscale und Web Push erreichbar ist. Wie würdest du das strukturieren?",
    )

    assert result["route"] == "direct"
    assert "Ordner erstellt" not in result["reply"]


def test_llm_coding_plan_does_not_override_strategic_direct_route(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    class FakePlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            from ai_hub.schemas.manager_plan import ManagerPlan

            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="User asks for architecture advice.",
                    decision="coding",
                    reason="Coding might help later.",
                    user_reply="Ich würde das in Manager, Coding-Agent und Research-Agent aufteilen.",
                    internal_task_for_worker="Create a starter workspace structure.",
                    approval_needed=False,
                ),
            }

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=FakePlanner())

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Ich möchte mit dir ein kleines Python-Projekt starten, das später über Tailscale und Web Push erreichbar ist. Wie würdest du das strukturieren?",
    )

    assert result["route"] == "direct"
    assert "Ordner erstellt" not in result["reply"]
    assert result["messages"][-1]["meta"]["llm_decision"] == "coding"
    assert result["messages"][-1]["meta"]["final_decision"] == "direct"


def test_delegated_reply_uses_stable_manager_prefix(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    class FakePlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            from ai_hub.schemas.manager_plan import ManagerPlan

            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Coding task.",
                    decision="coding",
                    reason="Needs a coding worker.",
                    user_reply="Langer, freier LLM-Text, der nicht als Prefix erscheinen soll.",
                    internal_task_for_worker="Create a file in the workspace.",
                    approval_needed=False,
                ),
            }

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=FakePlanner())

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Erstelle in deinem Workspace eine hallo.txt-Datei mit dem Inhalt Hallo",
    )

    assert result["route"] == "coding"
    assert result["reply"].startswith("Ich delegiere das an den Coding-Agenten.")
    assert "Langer, freier LLM-Text" not in result["reply"]
    assert result["messages"][-1]["meta"]["manager_source"] == "ollama"
    assert result["messages"][-1]["meta"]["llm_decision"] == "coding"
    assert result["messages"][-1]["meta"]["final_decision"] == "coding"


def test_internal_payload_is_normalized_to_english(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte erstelle in deinem Workspace eine hallo.txt-Datei mit dem Inhalt Hallo",
    )

    assistant_message = result["messages"][-1]
    internal_payload = assistant_message["meta"]["internal_payload"]
    assert internal_payload["language"] == "en"
    assert "Create" in internal_payload["task"] or "create" in internal_payload["task"]


def test_store_can_delete_thread(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore

    store = HubStore()
    thread = store.ensure_thread(None, "Wird geloescht")
    store.add_message(thread["id"], "user", "Hallo")

    deleted = store.delete_thread(thread["id"])

    assert deleted is True
    assert store.get_thread(thread["id"]) is None


def test_coding_agent_returns_structured_actions_and_summary(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent

    agent = CodingAgent()
    result = agent.handle_task(
        thread_id="thread-123",
        user_task="Erstelle einen Unterordner demo und darin notes.txt",
        history=[],
    )

    assert result["status"] == "completed"
    assert len(result["actions_executed"]) == 2
    assert [item["action_type"] for item in result["actions_executed"]] == ["make_directory", "create_file"]
    assert result["actions_blocked"] == []
    assert result["approval_request_created"] is False
    assert "Executed actions: make_directory, create_file." in result["internal_summary"]


def test_coding_agent_executes_create_then_request_execution_in_order(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent

    agent = CodingAgent()
    result = agent.handle_task(
        thread_id="thread-abc",
        user_task="Bitte erstelle eine kleine Python-Datei, die Hello World ausgibt, und beantrage anschließend die Ausführung.",
        history=[],
    )

    assert result["status"] == "completed_with_approval"
    assert [item["action_type"] for item in result["actions_executed"]] == [
        "make_directory",
        "create_file",
        "request_execution",
    ]
    assert result["approval_request_created"] is True
    assert result["approval_request"] is not None


def test_coding_agent_prefers_structured_plan_over_text(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.coding_agent import CodingAgent
    from ai_hub.schemas.coding_actions import CodingActionBatch, CreateFileAction
    from ai_hub.schemas.coding_delegation import CodingDelegationPlan
    from ai_hub.tools.file_tools import read_file

    agent = CodingAgent()
    structured_plan = CodingDelegationPlan(
        summary="Use a structured file action.",
        rationale="Manager provided an explicit action batch.",
        actions=CodingActionBatch(
            actions=[CreateFileAction(path="demo/structured.txt", content="OK", content_inferred=False)]
        ),
    )

    result = agent.handle_task(
        thread_id="thread-structured",
        user_task="Bitte lies nur etwas vor.",
        history=[],
        structured_plan=structured_plan,
    )

    assert result["status"] == "completed"
    assert result["actions_executed"][0]["action_type"] == "create_file"
    assert read_file("thread-structured", "demo/structured.txt") == "OK"


def test_workflow_stores_structured_coding_plan_metadata(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    class FakePlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            from ai_hub.schemas.coding_actions import CodingActionBatch, CreateFileAction
            from ai_hub.schemas.coding_delegation import CodingDelegationPlan
            from ai_hub.schemas.manager_plan import ManagerPlan

            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Create a file in coding.",
                    decision="coding",
                    reason="Needs coding action.",
                    user_reply="Ich übernehme die Vorbereitung.",
                    internal_task_for_worker="Create the requested file in the workspace.",
                    approval_needed=False,
                    coding_plan=CodingDelegationPlan(
                        summary="Structured coding plan",
                        rationale="Planner produced a direct create-file action.",
                        actions=CodingActionBatch(
                            actions=[CreateFileAction(path="src/from_plan.py", content='print("Hi")\n')]
                        ),
                    ),
                ),
            }

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=FakePlanner())
    result = workflow.handle_chat(thread_id=None, user_message="Bitte erstelle eine Python-Datei.")

    meta = result["messages"][-1]["meta"]
    assert result["route"] == "coding"
    assert meta["structured_plan_used"] is True
    assert meta["structured_action_count"] == 1
    assert meta["plan_validation_success"] is True
