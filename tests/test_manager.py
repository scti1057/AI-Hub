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


class FakeCodingClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, model: str, prompt: str, temperature: float = 0.2) -> str:
        return self.response


class RecordingWorker:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[dict] = []

    def handle_task(
        self,
        thread_id: str,
        user_task: str,
        history: list[dict],
        internal_task: str | None = None,
        structured_plan=None,
    ) -> dict:
        self.calls.append(
            {
                "thread_id": thread_id,
                "user_task": user_task,
                "history": history,
                "internal_task": internal_task,
                "structured_plan": structured_plan,
            }
        )
        payload = dict(self.result)
        payload.setdefault(
            "internal_payload",
            {
                "language": "en",
                "thread_id": thread_id,
                "task": internal_task or user_task,
            },
        )
        payload.setdefault("internal_summary", "")
        payload.setdefault("tool_results", [])
        return payload


class DeterministicPlanner:
    def plan(self, history_text: str, user_message: str, user_language: str):
        from ai_hub.agents.coding_agent import CodingAgent
        from ai_hub.schemas.coding_delegation import CodingDelegationPlan
        from ai_hub.schemas.manager_plan import ManagerPlan

        lowered = user_message.lower()
        if any(term in lowered for term in ("recherche", "research", "analyse")):
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Research task.",
                    decision="research",
                    reason="Needs analytical review.",
                    user_reply="Ich delegiere die Analyse.",
                    internal_task_for_worker="Analyze the request and summarize open questions.",
                    approval_needed=False,
                ),
            }
        if any(term in lowered for term in ("implementierungsplan", "roadmap", "arbeitspakete", "milestones")):
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Planning task.",
                    decision="plan",
                    reason="The user wants planning before implementation.",
                    user_reply="Ich bereite zuerst einen Projektplan vor.",
                    internal_task_for_worker="Prepare a project plan before implementation starts.",
                    approval_needed=False,
                ),
            }
        if any(term in lowered for term in ("kritisch", "zweitmeinung", "review", "prüfe", "pruefe")):
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Review task.",
                    decision="review",
                    reason="Needs critical review.",
                    user_reply="Ich delegiere die Prüfung.",
                    internal_task_for_worker="Review the latest plan critically and identify risks.",
                    approval_needed=False,
                ),
            }
        if any(term in lowered for term in ("datei", "workspace", "ordner", "pytest", ".py", "ausführung", "ausfuehrung", "lösche", "loesche", "lies")):
            batch = CodingAgent().build_action_batch_from_text(user_message)
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Coding task.",
                    decision="coding",
                    reason="Needs workspace action.",
                    user_reply="Ich delegiere das Coding.",
                    internal_task_for_worker="Execute the requested workspace actions.",
                    approval_needed=any(action.action_type == "request_execution" for action in batch.actions),
                    coding_plan=CodingDelegationPlan(
                        summary="Structured coding plan for tests.",
                        rationale="Derived deterministically inside the test planner.",
                        approval_needed=any(action.action_type == "request_execution" for action in batch.actions),
                        actions=batch,
                    ),
                ),
            }
        return {
            "enabled": True,
            "source": "ollama",
            "plan": ManagerPlan(
                summary="Direct manager reply.",
                decision="direct",
                reason="No worker is required.",
                user_reply="LLM-Antwort: Ich kann koordinieren und Freigaben beachten.",
                internal_task_for_worker="",
                approval_needed=False,
            ),
        }


def test_manager_creates_thread_and_persists_messages(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

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
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

    result = workflow.handle_chat(thread_id=None, user_message="Test Chat")

    assert result["route"] == "direct"
    assert result["approval_request"] is None


def test_manager_creates_pending_approval_for_python_execution(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte teste dieses Python-Skript mit pytest.",
    )

    approval = result["approval_request"]
    assert approval is not None
    assert approval["status"] == ApprovalStatus.PENDING.value
    assert approval["tool_name"] == "request_python_execution"


def test_code_runner_accepts_pytest_flags_and_workspace_test_paths(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.tools import file_tools
    from ai_hub.tools.code_runner import build_python_execution_request, validate_python_execution_request

    thread_id = "thread-pytest-flags"
    file_tools.write_file(thread_id, "tests/test_game_engine.py", "def test_ok():\n    assert True\n")

    command = build_python_execution_request(
        thread_id=thread_id,
        argv=["-m", "pytest", "tests/test_game_engine.py", "-v"],
        rationale="Run pytest for one workspace test file.",
    )
    details = validate_python_execution_request(command)

    assert details["argv"] == ["-m", "pytest", "tests/test_game_engine.py", "-v"]


def test_workspace_venv_bootstraps_pip_when_missing(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    import ai_hub.tools.code_runner as code_runner

    workspace = tmp_path / "workspaces" / "thread-pip-bootstrap"
    venv_path = workspace / ".venv"
    python_path = venv_path / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("", encoding="utf-8")

    calls: list[list[str]] = []

    class Completed:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if list(cmd[1:4]) == ["-m", "pip", "--version"]:
            if sum(1 for call in calls if call[1:4] == ["-m", "pip", "--version"]) == 1:
                return Completed(1, stderr="No module named pip")
            return Completed(0, stdout="pip 25.0")
        if list(cmd[1:4]) == ["-m", "ensurepip", "--upgrade"]:
            return Completed(0, stdout="installed")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(code_runner.subprocess, "run", fake_run)

    result = code_runner._ensure_workspace_venv(venv_path)

    assert result == python_path.resolve()
    assert [call[1:4] for call in calls] == [
        ["-m", "pip", "--version"],
        ["-m", "ensurepip", "--upgrade"],
        ["-m", "pip", "--version"],
    ]


def test_manager_executes_allowed_workspace_write(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.tools.file_tools import read_file

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

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
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

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


def test_manager_requests_dependency_install_for_missing_workspace_package(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.tools.file_tools import write_file

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())
    thread = store.ensure_thread(None)

    write_file(thread["id"], "src/main.py", "import requests\n")
    workflow.delegation.coding_agent = RecordingWorker(
        {
            "status": "completed",
            "reply": "File created: `src/main.py`.",
            "user_reply": "File created: `src/main.py`.",
            "internal_summary": "Executed actions: create_file. Blocked actions: none. Approval request created: no.",
            "actions_executed": [{"action_type": "create_file", "target": "src/main.py"}],
            "actions_blocked": [],
            "workspace": str(tmp_path / "workspaces" / thread["id"]),
            "internal_payload": {
                "language": "en",
                "thread_id": thread["id"],
                "task": "Create src/main.py.",
                "self_check": {
                    "inspected_files": [],
                    "touched_paths": ["src/main.py"],
                    "follow_up": [],
                    "summary": "Inspected files after execution: src/main.py.",
                },
            },
        }
    )

    result = workflow.handle_chat(
        thread_id=thread["id"],
        user_message="Please create src/main.py",
    )

    approval = result["approval_request"]
    assert approval is not None
    assert approval["tool_name"] == "install_python_requirements"
    assert approval["command"]["kind"] == "pip_install"
    assert "requests" in approval["command"]["packages"]
    assert "requirements.txt" in result["reply"]

    dependency_status = store.get_artifact(thread["id"], "dependency_status")
    assert dependency_status is not None
    assert "requests" in dependency_status["content"]["missing_requirements"]


def test_manager_creates_nested_directory_and_file(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.tools.file_tools import read_file

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

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
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

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
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte lies /etc/passwd und vergleiche es mit ../secrets.txt.",
    )

    assert result["route"] == "coding"
    assert "Blockiert:" in result["reply"]
    assert result["approval_request"] is None


def test_manager_persists_thread_artifacts_for_research(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte mache eine Recherche zur Architektur und fasse die Risiken zusammen.",
    )

    artifacts = store.list_artifacts(result["thread"]["id"])
    assert any(artifact["kind"] == "research_notes" for artifact in artifacts)
    research_artifact = next(artifact for artifact in artifacts if artifact["kind"] == "research_notes")
    assert research_artifact["summary"]
    assert research_artifact["content"]["status"] == "completed"


def test_manager_consults_research_and_review_before_large_coding_task(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.schemas.coding_actions import CodingActionBatch, CreateFileAction
    from ai_hub.schemas.coding_delegation import CodingDelegationPlan
    from ai_hub.schemas.manager_plan import ManagerPlan

    class StrategicCodingPlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Large feature request that should be decomposed before implementation starts.",
                    decision="coding",
                    reason="Needs coding work.",
                    user_reply="Ich delegiere das Coding.",
                    internal_task_for_worker="Implement the requested project incrementally.",
                    approval_needed=False,
                    coding_plan=CodingDelegationPlan(
                        summary="Start with one bootstrap file.",
                        rationale="Keep the first slice intentionally small.",
                        approval_needed=False,
                        actions=CodingActionBatch(
                            actions=[CreateFileAction(path="src/main.py", content="", content_inferred=False)]
                        ),
                    ),
                ),
            }

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=StrategicCodingPlanner())
    research_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Research reply",
            "user_reply": "Research reply",
            "internal_summary": "Task decomposed into setup, logic, tests, and UI.",
        }
    )
    review_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Review reply",
            "user_reply": "Review reply",
            "internal_summary": "The first slice is small enough but should keep scope limited to bootstrap code.",
        }
    )
    coding_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Datei erstellt: `src/main.py`.",
            "user_reply": "Datei erstellt: `src/main.py`.",
            "internal_summary": "Executed actions: create_file. Blocked actions: none. Approval request created: no.",
            "actions_executed": [{"action_type": "create_file", "target": "src/main.py"}],
            "actions_blocked": [],
            "workspace": str(tmp_path / "workspaces" / "thread"),
        }
    )
    workflow.delegation.research_agent = research_worker
    workflow.delegation.reviewer_agent = review_worker
    workflow.delegation.coding_agent = coding_worker

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte strukturiere ein größeres TicTacToe-Projekt mit GUI, Spiellogik, Tests und sauberer Architektur und lege dafür die ersten Dateien an.",
    )

    assert result["route"] == "coding"
    assert len(research_worker.calls) == 1
    assert len(review_worker.calls) == 2
    assert "Research summary:" in coding_worker.calls[0]["internal_task"]
    assert "Reviewer summary:" in coding_worker.calls[0]["internal_task"]
    assert "Ich habe die Aufgabe zuerst intern in kleinere Schritte und Risiken zerlegt." in result["reply"]
    assert "Ich habe den letzten Coding-Schritt zusätzlich intern gegenprüfen lassen." in result["reply"]

    artifacts = {artifact["kind"]: artifact for artifact in store.list_artifacts(result["thread"]["id"])}
    assert "research_notes" in artifacts
    assert "review_notes" in artifacts
    assert "project_brief" in artifacts
    assert "coding_status" in artifacts
    assert "implementation_review" in artifacts


def test_manager_uses_project_plan_route_and_stores_project_state(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())
    research_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Research reply",
            "user_reply": "Research reply",
            "internal_summary": "A phased project plan is feasible and should start with a thin playable slice.",
            "internal_payload": {
                "language": "en",
                "thread_id": "ignored",
                "task": "research: project plan",
                "summary": "A phased project plan is feasible and should start with a thin playable slice.",
                "findings": [
                    "Use a small repo layout with src, tests, and a short README.",
                    "Start with game logic before polishing the GUI.",
                ],
                "assumptions": ["Tkinter is acceptable for the first GUI iteration."],
                "open_questions": ["Should the first version support local multiplayer only?"],
                "recommendation": "Start with one playable local version and confirm the GUI scope with the user.",
                "sources": [],
            },
        }
    )
    review_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Review reply",
            "user_reply": "Review reply",
            "internal_summary": "The project plan is reasonable, but implementation should wait for user confirmation on scope.",
            "internal_payload": {
                "language": "en",
                "thread_id": "ignored",
                "task": "review: project plan",
                "summary": "The project plan is reasonable, but implementation should wait for user confirmation on scope.",
                "findings": [
                    "Do not start with a single-file prototype if the user clearly wants a project structure.",
                ],
                "assumptions": [],
                "open_questions": ["Should tests be included in the first implementation slice?"],
                "recommendation": "Return to the user with the phased plan before broad implementation starts.",
            },
        }
    )
    workflow.delegation.research_agent = research_worker
    workflow.delegation.reviewer_agent = review_worker

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte erstelle zuerst einen Implementierungsplan mit Arbeitspaketen und Milestones für ein größeres TicTacToe-Projekt.",
    )

    assert result["route"] == "plan"
    assert "Projektplan" in result["reply"] or "Projekt" in result["reply"]
    artifacts = {artifact["kind"]: artifact for artifact in store.list_artifacts(result["thread"]["id"])}
    assert "project_plan" in artifacts
    assert "project_state" in artifacts
    assert artifacts["project_state"]["content"]["awaiting_user_feedback"] is True
    assert artifacts["project_plan"]["content"]["next_steps"]


def test_manager_resumes_project_from_user_feedback(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.schemas.manager_plan import ManagerPlan

    class PlanningThenDirectPlanner:
        def __init__(self) -> None:
            self.calls = 0

        def plan(self, history_text: str, user_message: str, user_language: str):
            self.calls += 1
            if self.calls == 1:
                return {
                    "enabled": True,
                    "source": "ollama",
                    "plan": ManagerPlan(
                        summary="Planning task.",
                        decision="plan",
                        reason="The user wants a plan first.",
                        user_reply="Ich bereite zuerst einen Projektplan vor.",
                        internal_task_for_worker="Prepare a project plan before implementation starts.",
                        approval_needed=False,
                        coding_plan=None,
                    ),
                }
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Direct task.",
                    decision="direct",
                    reason="A direct reply would normally be enough.",
                    user_reply="Direkte Antwort.",
                    internal_task_for_worker="",
                    approval_needed=False,
                    coding_plan=None,
                ),
            }

    store = HubStore()
    planner = PlanningThenDirectPlanner()
    workflow = ManagerWorkflow(store=store, planner=planner)
    research_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Research reply",
            "user_reply": "Research reply",
            "internal_summary": "A phased plan is feasible.",
            "internal_payload": {
                "language": "en",
                "thread_id": "ignored",
                "task": "research",
                "summary": "A phased plan is feasible.",
                "findings": ["Use src, tests, and README."],
                "assumptions": [],
                "open_questions": ["Should tests be included in the first slice?"],
                "recommendation": "Start with a small playable first slice.",
                "sources": [],
            },
        }
    )
    review_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Review reply",
            "user_reply": "Review reply",
            "internal_summary": "Wait for user confirmation before implementation.",
            "internal_payload": {
                "language": "en",
                "thread_id": "ignored",
                "task": "review",
                "summary": "Wait for user confirmation before implementation.",
                "findings": ["Do not start with one giant file."],
                "assumptions": [],
                "open_questions": [],
                "recommendation": "After approval, implement the first small slice only.",
            },
        }
    )
    coding_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Datei erstellt: `tictactoe/src/main.py`.",
            "user_reply": "Datei erstellt: `tictactoe/src/main.py`.",
            "internal_summary": "Executed actions: create_file.",
            "actions_executed": [{"action_type": "create_file", "target": "tictactoe/src/main.py"}],
            "actions_blocked": [],
            "workspace": str(tmp_path / "workspaces" / "thread"),
            "internal_payload": {
                "language": "en",
                "thread_id": "ignored",
                "task": "Continue the active project using the previously approved plan.",
                "self_check": {
                    "inspected_files": [{"path": "tictactoe/src/main.py", "preview": "print('ok')\n", "bytes": 12}],
                    "touched_paths": ["tictactoe/src/main.py"],
                    "follow_up": [],
                    "summary": "Inspected files after execution: tictactoe/src/main.py.",
                },
            },
        }
    )
    workflow.delegation.research_agent = research_worker
    workflow.delegation.reviewer_agent = review_worker
    workflow.delegation.coding_agent = coding_worker

    plan_result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte erstelle zuerst einen Implementierungsplan mit Arbeitspaketen und Milestones.",
    )
    follow_up = workflow.handle_chat(
        thread_id=plan_result["thread"]["id"],
        user_message="Das passt so, bitte mach weiter und setze den ersten kleinen Schritt um.",
    )

    assert plan_result["route"] == "plan"
    assert follow_up["route"] == "coding"
    assert len(coding_worker.calls) == 1
    assert len(research_worker.calls) == 1
    assert len(review_worker.calls) == 2
    assert "Continue the active project" in coding_worker.calls[0]["internal_task"]
    artifacts = {artifact["kind"]: artifact for artifact in store.list_artifacts(plan_result["thread"]["id"])}
    assert artifacts["project_state"]["content"]["phase"] == "implementation"


def test_manager_runs_multiple_autonomous_project_steps_in_one_turn(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.schemas.coding_actions import CodingActionBatch, CreateFileAction
    from ai_hub.schemas.coding_delegation import CodingDelegationPlan
    from ai_hub.schemas.manager_plan import ManagerPlan, ProjectOutline

    class AutonomousPlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Build the requested project in small steps.",
                    decision="coding",
                    reason="A bounded autonomous project run is appropriate.",
                    user_reply="I will implement the project in small steps and stop when it is ready to test or needs approval.",
                    internal_task_for_worker="Build the requested project with a clean repository structure.",
                    approval_needed=False,
                    coding_plan=CodingDelegationPlan(
                        summary="Create the first repository file.",
                        rationale="The first step creates real structure before follow-up steps continue.",
                        approval_needed=False,
                        actions=CodingActionBatch(
                            actions=[
                                CreateFileAction(
                                    path="src/game_logic.py",
                                    content="def winner(board):\n    return None\n",
                                    content_inferred=False,
                                )
                            ]
                        ),
                    ),
                    project_outline=ProjectOutline(
                        summary="Tic-tac-toe should be built as a small repository with a smoke-test step.",
                        repo_structure=[
                            "src/game_logic.py - core rules",
                            "src/main.py - entry point",
                        ],
                        steps=[
                            "Create the repository scaffold and the core game logic module.",
                            "Wire the main entry point to the game logic.",
                        ],
                        validation_steps=[
                            "Request one smoke test execution for src/main.py.",
                        ],
                        completion_criteria=[
                            "The project can be started from src/main.py without obvious import errors.",
                            "The first playable scenario is ready to test.",
                        ],
                        autonomous_execution=True,
                    ),
                ),
            }

    class SequentialCodingWorker:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.results = [
                {
                    "status": "completed",
                    "reply": "File created: `src/game_logic.py`.",
                    "user_reply": "File created: `src/game_logic.py`.",
                    "internal_summary": "Executed actions: create_file. Blocked actions: none. Approval request created: no. Self-check: Inspected files after execution: src/game_logic.py.",
                    "actions_executed": [{"action_type": "create_file", "target": "src/game_logic.py"}],
                    "actions_blocked": [],
                    "workspace": str(tmp_path / "workspaces" / "thread"),
                    "internal_payload": {
                        "language": "en",
                        "thread_id": "ignored",
                        "task": "step 1",
                        "self_check": {
                            "inspected_files": [{"path": "src/game_logic.py", "preview": "def winner(board):\n    return None\n", "bytes": 35}],
                            "touched_paths": ["src/game_logic.py"],
                            "follow_up": [],
                            "summary": "Inspected files after execution: src/game_logic.py.",
                        },
                    },
                },
                {
                    "status": "completed",
                    "reply": "File created: `src/main.py`.",
                    "user_reply": "File created: `src/main.py`.",
                    "internal_summary": "Executed actions: create_file. Blocked actions: none. Approval request created: no. Self-check: Inspected files after execution: src/main.py.",
                    "actions_executed": [{"action_type": "create_file", "target": "src/main.py"}],
                    "actions_blocked": [],
                    "workspace": str(tmp_path / "workspaces" / "thread"),
                    "internal_payload": {
                        "language": "en",
                        "thread_id": "ignored",
                        "task": "step 2",
                        "self_check": {
                            "inspected_files": [{"path": "src/main.py", "preview": "from game_logic import winner\nprint('ready')\n", "bytes": 41}],
                            "touched_paths": ["src/main.py"],
                            "follow_up": [],
                            "summary": "Inspected files after execution: src/main.py.",
                        },
                    },
                },
                {
                    "status": "completed_with_approval",
                    "reply": "Approval required: `python src/main.py` has been prepared as an execution request.",
                    "user_reply": "Approval required: `python src/main.py` has been prepared as an execution request.",
                    "internal_summary": "Executed actions: request_execution. Blocked actions: none. Approval request created: yes. Self-check: Follow-up: Runtime validation is still pending user approval.",
                    "actions_executed": [{"action_type": "request_execution", "target": "src/main.py"}],
                    "actions_blocked": [],
                    "workspace": str(tmp_path / "workspaces" / "thread"),
                    "approval_request": {
                        "tool_name": "request_python_execution",
                        "command": {
                            "thread_id": "thread-autonomous",
                            "argv": ["src/main.py"],
                            "preview": "python src/main.py",
                            "rationale": "Smoke test the entry point.",
                        },
                        "rationale": "Smoke test the entry point.",
                    },
                    "internal_payload": {
                        "language": "en",
                        "thread_id": "ignored",
                        "task": "validation",
                        "self_check": {
                            "inspected_files": [{"path": "src/main.py", "preview": "from game_logic import winner\nprint('ready')\n", "bytes": 41}],
                            "touched_paths": ["src/main.py"],
                            "follow_up": ["Runtime validation is still pending user approval."],
                            "summary": "Follow-up: Runtime validation is still pending user approval.",
                        },
                    },
                },
            ]

        def handle_task(self, thread_id: str, user_task: str, history: list[dict], internal_task: str | None = None, structured_plan=None) -> dict:
            self.calls.append(
                {
                    "thread_id": thread_id,
                    "user_task": user_task,
                    "history": history,
                    "internal_task": internal_task,
                    "structured_plan": structured_plan,
                }
            )
            return dict(self.results[len(self.calls) - 1])

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=AutonomousPlanner())
    coding_worker = SequentialCodingWorker()
    review_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Review reply",
            "user_reply": "Review reply",
            "internal_summary": "Run one focused smoke test before calling the project ready to test.",
        }
    )
    workflow.delegation.coding_agent = coding_worker
    workflow.delegation.reviewer_agent = review_worker

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Please build a small tic-tac-toe project with a real repository structure and stop when it is ready to test.",
    )

    assert result["route"] == "coding"
    assert len(coding_worker.calls) == 3
    assert "Current step 1 of 3" in coding_worker.calls[0]["internal_task"]
    assert "Current step 2 of 3" in coding_worker.calls[1]["internal_task"]
    assert "Current step 3 of 3" in coding_worker.calls[2]["internal_task"]
    assert result["approval_request"] is not None
    assert "autonomously" in result["reply"].lower()

    artifacts = {artifact["kind"]: artifact for artifact in store.list_artifacts(result["thread"]["id"])}
    project_state = artifacts["project_state"]["content"]
    assert project_state["autonomous_mode"] is True
    assert project_state["pending_step"] == "Request one smoke test execution for src/main.py."
    assert len(project_state["completed_steps"]) == 2
    assert project_state["ready_to_test"] is False


def test_approve_execution_resumes_autonomous_project_run(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.tools.file_tools import write_file

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())
    thread = store.ensure_thread(None)
    write_file(thread["id"], "src/main.py", "print('ready')\n")

    store.upsert_artifact(
        thread_id=thread["id"],
        kind="project_plan",
        title="Project Plan",
        summary="Autonomous project plan",
        content={
            "user_message": "Continue project",
            "summary": "Autonomous project plan",
            "repo_structure": ["src/main.py - entry point"],
            "steps": ["Create the entry point.", "Run one smoke test for src/main.py.", "Mark the project ready to test."],
            "validation_steps": ["Run one smoke test for src/main.py."],
            "completion_criteria": ["The entry point runs successfully."],
            "project_outline": {
                "summary": "Autonomous project plan",
                "repo_structure": ["src/main.py - entry point"],
                "steps": ["Create the entry point.", "Run one smoke test for src/main.py.", "Mark the project ready to test."],
                "validation_steps": [],
                "completion_criteria": ["The entry point runs successfully."],
                "autonomous_execution": True,
            },
        },
    )
    store.upsert_artifact(
        thread_id=thread["id"],
        kind="project_state",
        title="Project State",
        summary="Waiting for approved execution.",
        content={
            "phase": "awaiting_approval",
            "awaiting_user_feedback": False,
            "autonomous_mode": True,
            "ready_to_test": False,
            "last_user_request": "Continue project",
            "last_summary": "Waiting for approved execution.",
            "latest_status": "approval_required",
            "steps": ["Create the entry point.", "Run one smoke test for src/main.py.", "Mark the project ready to test."],
            "completed_steps": ["Create the entry point."],
            "pending_step": "Run one smoke test for src/main.py.",
            "remaining_steps": ["Mark the project ready to test."],
            "next_steps": ["Mark the project ready to test."],
        },
    )

    approval = store.create_approval_request(
        thread_id=thread["id"],
        agent_role="coding",
        tool_name="request_python_execution",
        command={
            "thread_id": thread["id"],
            "argv": ["src/main.py"],
            "preview": "python src/main.py",
            "rationale": "Smoke test the entry point.",
        },
        rationale="Smoke test the entry point.",
    )

    coding_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Project is ready to test.",
            "user_reply": "Project is ready to test.",
            "internal_summary": "Executed actions: none. Blocked actions: none. Approval request created: no. Self-check: No immediate follow-up risk was detected from the executed action batch.",
            "actions_executed": [],
            "actions_blocked": [],
            "workspace": str(tmp_path / "workspaces" / thread["id"]),
            "internal_payload": {
                "language": "en",
                "thread_id": thread["id"],
                "task": "Finalize ready-to-test state.",
                "self_check": {
                    "inspected_files": [],
                    "touched_paths": [],
                    "follow_up": [],
                    "summary": "No immediate follow-up risk was detected from the executed action batch.",
                },
            },
        }
    )
    workflow.delegation.coding_agent = coding_worker
    workflow.delegation.reviewer_agent = RecordingWorker(
        {
            "status": "completed",
            "reply": "Review reply",
            "user_reply": "Review reply",
            "internal_summary": "The ready-to-test handoff is acceptable.",
        }
    )

    result = workflow.approve_execution(approval["id"])

    assert result["status"] == "executed"
    assert result.get("continuation") is not None
    assert len(coding_worker.calls) == 1

    project_state = store.get_artifact(thread["id"], "project_state")["content"]
    assert project_state["ready_to_test"] is True
    assert project_state["pending_step"] is None
    assert len(project_state["completed_steps"]) == 3
    assert project_state["awaiting_user_feedback"] is True


def test_failed_execution_triggers_autonomous_debug_repair_loop(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())
    thread = store.ensure_thread(None)

    store.upsert_artifact(
        thread_id=thread["id"],
        kind="project_plan",
        title="Project Plan",
        summary="Autonomous project plan",
        content={
            "project_outline": {
                "summary": "Autonomous project plan",
                "repo_structure": ["src/main.py - entry point"],
                "steps": ["Create the entry point.", "Run one smoke test for src/main.py.", "Mark the project ready to test."],
                "validation_steps": [],
                "completion_criteria": ["The entry point runs successfully."],
                "autonomous_execution": True,
            },
        },
    )
    store.upsert_artifact(
        thread_id=thread["id"],
        kind="project_state",
        title="Project State",
        summary="Waiting for approved execution.",
        content={
            "phase": "awaiting_approval",
            "awaiting_user_feedback": False,
            "autonomous_mode": True,
            "ready_to_test": False,
            "last_user_request": "Continue project",
            "last_summary": "Waiting for approved execution.",
            "latest_status": "approval_required",
            "steps": ["Create the entry point.", "Run one smoke test for src/main.py.", "Mark the project ready to test."],
            "completed_steps": ["Create the entry point."],
            "pending_step": "Run one smoke test for src/main.py.",
            "remaining_steps": ["Mark the project ready to test."],
            "next_steps": ["Mark the project ready to test."],
            "debug_attempts": 0,
        },
    )

    approval = store.create_approval_request(
        thread_id=thread["id"],
        agent_role="coding",
        tool_name="request_python_execution",
        command={
            "thread_id": thread["id"],
            "argv": ["src/main.py"],
            "preview": "python src/main.py",
            "rationale": "Smoke test the entry point.",
        },
        rationale="Smoke test the entry point.",
    )

    monkeypatch.setattr(
        "ai_hub.orchestration.workflow.execute_python_approval",
        lambda command: {
            "command": ["python", "src/main.py"],
            "command_kind": "python_execution",
            "returncode": 1,
            "stdout": "",
            "stderr": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'missing_pkg'",
            "workspace": str(tmp_path / "workspaces" / thread["id"]),
            "sandbox_backend": "subprocess",
        },
    )

    coding_worker = RecordingWorker(
        {
            "status": "completed_with_approval",
            "reply": "Prepared another execution request.",
            "user_reply": "Prepared another execution request.",
            "internal_summary": "Executed actions: create_file, request_execution. Blocked actions: none. Approval request created: yes.",
            "actions_executed": [
                {"action_type": "create_file", "target": "src/main.py"},
                {"action_type": "request_execution", "target": "src/main.py"},
            ],
            "actions_blocked": [],
            "workspace": str(tmp_path / "workspaces" / thread["id"]),
            "approval_request": {
                "tool_name": "request_python_execution",
                "command": {
                    "thread_id": thread["id"],
                    "argv": ["src/main.py"],
                    "preview": "python src/main.py",
                    "rationale": "Retry after repair.",
                },
                "rationale": "Retry after repair.",
            },
            "internal_payload": {
                "language": "en",
                "thread_id": thread["id"],
                "task": "Repair the entry point after the failed run.",
                "self_check": {
                    "inspected_files": [],
                    "touched_paths": ["src/main.py"],
                    "follow_up": ["Runtime validation is still pending user approval."],
                    "summary": "Follow-up: Runtime validation is still pending user approval.",
                },
            },
        }
    )
    workflow.delegation.coding_agent = coding_worker
    workflow.delegation.reviewer_agent = RecordingWorker(
        {
            "status": "completed",
            "reply": "Review reply",
            "user_reply": "Review reply",
            "internal_summary": "The repair direction is reasonable.",
        }
    )

    result = workflow.approve_execution(approval["id"])

    assert result["status"] == "failed"
    assert result.get("continuation") is not None
    assert len(coding_worker.calls) == 1
    assert "ModuleNotFoundError" in coding_worker.calls[0]["internal_task"]
    assert "Current step 2 of 3: Run one smoke test for src/main.py." in coding_worker.calls[0]["internal_task"]

    project_state = store.get_artifact(thread["id"], "project_state")["content"]
    assert project_state["phase"] == "awaiting_approval"
    assert project_state["debug_attempts"] == 1


def test_user_feedback_resumes_blocked_autonomous_project_from_current_state(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())
    thread = store.ensure_thread(None)

    store.upsert_artifact(
        thread_id=thread["id"],
        kind="project_plan",
        title="Project Plan",
        summary="Stored project plan.",
        content={
            "summary": "Stored project plan.",
            "repo_structure": ["src/game_logic.py", "src/gui.py", "src/main.py"],
            "steps": [
                "Implement core game logic.",
                "Develop the GUI layout.",
                "Implement the 1v1 game loop.",
            ],
            "validation_steps": [],
            "completion_criteria": ["The project is ready for a first test run."],
            "project_outline": {
                "summary": "Stored project plan.",
                "repo_structure": ["src/game_logic.py", "src/gui.py", "src/main.py"],
                "steps": [
                    "Implement core game logic.",
                    "Develop the GUI layout.",
                    "Implement the 1v1 game loop.",
                ],
                "validation_steps": [],
                "completion_criteria": ["The project is ready for a first test run."],
                "autonomous_execution": True,
            },
        },
    )
    store.upsert_artifact(
        thread_id=thread["id"],
        kind="project_state",
        title="Project State",
        summary="Implementation blocked after the GUI step.",
        content={
            "phase": "blocked",
            "awaiting_user_feedback": True,
            "autonomous_mode": True,
            "ready_to_test": False,
            "last_user_request": "Continue project",
            "last_summary": "Implementation blocked after the GUI step.",
            "latest_status": "blocked",
            "repo_structure": ["src/game_logic.py", "src/gui.py", "src/main.py"],
            "steps": [
                "Implement core game logic.",
                "Develop the GUI layout.",
                "Implement the 1v1 game loop.",
            ],
            "completed_steps": [
                "Implement core game logic.",
                "Develop the GUI layout.",
            ],
            "pending_step": None,
            "remaining_steps": ["Implement the 1v1 game loop."],
            "next_steps": ["Implement the 1v1 game loop."],
            "last_execution_result": {
                "command": ["python", "src/main.py"],
                "returncode": 1,
                "stdout": "",
                "stderr": "Traceback ... NameError: name 'TicTacToeGame' is not defined",
            },
        },
    )

    coding_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Implemented the next step.",
            "user_reply": "Implemented the next step.",
            "internal_summary": "Executed actions: create_file. Blocked actions: none. Approval request created: no.",
            "actions_executed": [{"action_type": "create_file", "target": "src/main.py"}],
            "actions_blocked": [],
            "workspace": str(tmp_path / "workspaces" / thread["id"]),
            "internal_payload": {
                "language": "en",
                "thread_id": thread["id"],
                "task": "Continue from the blocked project state.",
                "self_check": {
                    "inspected_files": [],
                    "touched_paths": ["src/main.py"],
                    "follow_up": [],
                    "summary": "No immediate follow-up risk was detected.",
                },
            },
        }
    )
    workflow.delegation.coding_agent = coding_worker
    workflow.delegation.reviewer_agent = RecordingWorker(
        {
            "status": "completed",
            "reply": "Review reply",
            "user_reply": "Review reply",
            "internal_summary": "The continuation step looks acceptable.",
        }
    )

    result = workflow.handle_chat(
        thread_id=thread["id"],
        user_message="Please continue and keep building on the current project state.",
    )

    assert result["route"] == "coding"
    assert len(coding_worker.calls) == 1
    internal_task = coding_worker.calls[0]["internal_task"]
    assert "Project phase: blocked" in internal_task
    assert "Completed steps: ['Implement core game logic.', 'Develop the GUI layout.']" in internal_task
    assert "NameError" in internal_task
    assert "Current step 3 of 3: Implement the 1v1 game loop." in internal_task


def test_dependency_install_approval_retries_same_autonomous_step(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())
    thread = store.ensure_thread(None)

    store.upsert_artifact(
        thread_id=thread["id"],
        kind="project_plan",
        title="Project Plan",
        summary="Stored plan.",
        content={
            "project_outline": {
                "summary": "Build the app.",
                "steps": ["Implement the application shell.", "Run a smoke test."],
                "repo_structure": ["src/main.py"],
                "validation_steps": [],
                "completion_criteria": ["App starts without errors."],
                "autonomous_execution": True,
            },
        },
    )
    store.upsert_artifact(
        thread_id=thread["id"],
        kind="project_state",
        title="Project State",
        summary="Waiting for dependency approval.",
        content={
            "phase": "awaiting_approval",
            "awaiting_user_feedback": False,
            "autonomous_mode": True,
            "ready_to_test": False,
            "last_user_request": "Continue project",
            "last_summary": "Waiting for dependency approval.",
            "latest_status": "completed_with_approval",
            "steps": ["Implement the application shell.", "Run a smoke test."],
            "completed_steps": [],
            "pending_step": "Implement the application shell.",
            "remaining_steps": ["Run a smoke test."],
            "next_steps": ["Run a smoke test."],
        },
    )

    approval = store.create_approval_request(
        thread_id=thread["id"],
        agent_role="coding",
        tool_name="install_python_requirements",
        command={
            "kind": "pip_install",
            "thread_id": thread["id"],
            "packages": ["requests"],
            "preview": "python -m pip install requests",
            "rationale": "Install missing dependencies.",
        },
        rationale="Install missing dependencies.",
    )

    monkeypatch.setattr(
        "ai_hub.orchestration.workflow.execute_python_approval",
        lambda command: {
            "command": ["python", "-m", "pip", "install", "requests"],
            "command_kind": "pip_install",
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "workspace": str(tmp_path / "workspaces" / thread["id"]),
            "sandbox_backend": "subprocess",
            "packages": ["requests"],
        },
    )

    coding_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Implemented the application shell.",
            "user_reply": "Implemented the application shell.",
            "internal_summary": "Executed actions: create_file. Blocked actions: none. Approval request created: no.",
            "actions_executed": [{"action_type": "create_file", "target": "src/main.py"}],
            "actions_blocked": [],
            "workspace": str(tmp_path / "workspaces" / thread["id"]),
            "internal_payload": {
                "language": "en",
                "thread_id": thread["id"],
                "task": "Implement the application shell.",
                "self_check": {
                    "inspected_files": [],
                    "touched_paths": ["src/main.py"],
                    "follow_up": [],
                    "summary": "No immediate follow-up risk was detected.",
                },
            },
        }
    )
    workflow.delegation.coding_agent = coding_worker
    workflow.delegation.reviewer_agent = RecordingWorker(
        {
            "status": "completed",
            "reply": "Review reply",
            "user_reply": "Review reply",
            "internal_summary": "The step looks acceptable.",
        }
    )

    result = workflow.approve_execution(approval["id"])

    assert result["status"] == "executed"
    assert result.get("continuation") is not None
    assert len(coding_worker.calls) == 1
    assert "Current step 1 of 2: Implement the application shell." in coding_worker.calls[0]["internal_task"]

def test_planning_context_includes_project_memory(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.schemas.manager_plan import ManagerPlan

    class RecordingPlanner:
        def __init__(self) -> None:
            self.history_calls: list[str] = []

        def plan(self, history_text: str, user_message: str, user_language: str):
            self.history_calls.append(history_text)
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Direct reply.",
                    decision="direct",
                    reason="No worker needed.",
                    user_reply="Direkte Antwort.",
                    internal_task_for_worker="",
                    approval_needed=False,
                    coding_plan=None,
                ),
            }

    store = HubStore()
    planner = RecordingPlanner()
    workflow = ManagerWorkflow(store=store, planner=planner)
    thread = store.ensure_thread(None)
    store.upsert_artifact(
        thread_id=thread["id"],
        kind="project_brief",
        title="Project Brief",
        summary="Existing project memory summary.",
        content={"scope": "demo"},
    )

    workflow.handle_chat(thread_id=thread["id"], user_message="Wie ist der aktuelle Stand?")

    assert "Project memory:" in planner.history_calls[0]
    assert "Existing project memory summary." in planner.history_calls[0]


def test_manager_can_persist_english_preference_per_thread(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.schemas.manager_plan import ManagerPlan

    class PreferencePlanner:
        def __init__(self) -> None:
            self.user_languages: list[str] = []

        def plan(self, history_text: str, user_message: str, user_language: str):
            self.user_languages.append(user_language)
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Direct reply.",
                    decision="direct",
                    reason="No worker required.",
                    user_reply="English reply." if user_language == "en" else "Deutsche Antwort.",
                    internal_task_for_worker="",
                    approval_needed=False,
                ),
            }

    store = HubStore()
    planner = PreferencePlanner()
    workflow = ManagerWorkflow(store=store, planner=planner)

    first = workflow.handle_chat(thread_id=None, user_message="Bitte sprich ab jetzt nur noch auf Englisch mit mir.")
    second = workflow.handle_chat(thread_id=first["thread"]["id"], user_message="Wie ist der aktuelle Stand?")

    assert planner.user_languages == ["en", "en"]
    assert second["reply"] == "English reply."
    preferences = store.get_artifact(first["thread"]["id"], "manager_preferences")
    assert preferences["content"]["preferred_user_language"] == "en"


def test_manager_full_implementation_followup_runs_autonomous_steps_from_stored_plan(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.schemas.manager_plan import ManagerPlan

    class PlanThenCodingPlanner:
        def __init__(self) -> None:
            self.calls = 0

        def plan(self, history_text: str, user_message: str, user_language: str):
            self.calls += 1
            if self.calls == 1:
                return {
                    "enabled": True,
                    "source": "ollama",
                    "plan": ManagerPlan(
                        summary="Plan the project first.",
                        decision="plan",
                        reason="The user explicitly requested a step-by-step plan.",
                        user_reply="I will prepare the plan first.",
                        internal_task_for_worker="Prepare a step-by-step project plan.",
                        approval_needed=False,
                    ),
                }
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Start implementation.",
                    decision="coding",
                    reason="The user approved the plan and wants the full implementation.",
                    user_reply="I will start implementing now.",
                    internal_task_for_worker="Start implementing the approved project.",
                    approval_needed=False,
                ),
            }

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=PlanThenCodingPlanner())
    research_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Research reply",
            "user_reply": "Research reply",
            "internal_summary": "A small repo with engine, UI, AI, and entry point is the right shape.",
            "internal_payload": {
                "language": "en",
                "thread_id": "ignored",
                "task": "research",
                "summary": "A small repo with engine, UI, AI, and entry point is the right shape.",
                "findings": ["Separate engine, UI, AI, and entry point files."],
                "assumptions": [],
                "open_questions": [],
                "recommendation": "Implement engine first, then UI, then AI and final integration.",
                "sources": [],
            },
        }
    )
    review_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Review reply",
            "user_reply": "Review reply",
            "internal_summary": "The plan is sound. After approval, implementation can proceed step by step.",
            "internal_payload": {
                "language": "en",
                "thread_id": "ignored",
                "task": "review",
                "summary": "The plan is sound.",
                "findings": ["Keep the first implementation step small and coherent."],
                "assumptions": [],
                "open_questions": [],
                "recommendation": "Move through the stored steps and summarize the final result.",
            },
        }
    )
    coding_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "File created.",
            "user_reply": "File created.",
            "internal_summary": "Executed actions: create_file. Blocked actions: none. Approval request created: no. Self-check: Inspected files after execution: src/main.py.",
            "actions_executed": [{"action_type": "create_file", "target": "src/main.py"}],
            "actions_blocked": [],
            "workspace": str(tmp_path / "workspaces" / "thread"),
            "internal_payload": {
                "language": "en",
                "thread_id": "ignored",
                "task": "autonomous step",
                "self_check": {
                    "inspected_files": [{"path": "src/main.py", "preview": "print('ok')\n", "bytes": 12}],
                    "touched_paths": ["src/main.py"],
                    "follow_up": [],
                    "summary": "Inspected files after execution: src/main.py.",
                },
            },
        }
    )
    workflow.delegation.research_agent = research_worker
    workflow.delegation.reviewer_agent = review_worker
    workflow.delegation.coding_agent = coding_worker

    plan_result = workflow.handle_chat(
        thread_id=None,
        user_message="Please code a tic tac toe game with a gui interface. It should have a 1v1 mode and a ai mode. I want it as clean as possible with nice features. Give me a step by step implementation plan.",
    )
    follow_up = workflow.handle_chat(
        thread_id=plan_result["thread"]["id"],
        user_message="This looks good! Start the implementation and let me know when you are done. Also write me a quick summary of what features you implemented.",
    )

    assert follow_up["route"] == "coding"
    assert len(coding_worker.calls) >= 2
    assert "Current step 1 of" in coding_worker.calls[0]["internal_task"]
    assert "Current step 2 of" in coding_worker.calls[1]["internal_task"]
    assert "ready for a first test run" in follow_up["reply"]
    assert "Implemented steps:" in follow_up["reply"]


def test_manager_reviews_regular_coding_result_and_stores_self_check(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.schemas.coding_actions import CodingActionBatch, CreateFileAction
    from ai_hub.schemas.coding_delegation import CodingDelegationPlan
    from ai_hub.schemas.manager_plan import ManagerPlan

    class SimpleCodingPlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Small coding task.",
                    decision="coding",
                    reason="Needs a workspace change.",
                    user_reply="Ich delegiere das Coding.",
                    internal_task_for_worker="Create one starter file.",
                    approval_needed=False,
                    coding_plan=CodingDelegationPlan(
                        summary="Create one file.",
                        rationale="Simple deterministic slice.",
                        approval_needed=False,
                        actions=CodingActionBatch(
                            actions=[CreateFileAction(path="src/main.py", content="print('ok')\n", content_inferred=False)]
                        ),
                    ),
                ),
            }

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=SimpleCodingPlanner())
    coding_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Datei erstellt: `src/main.py`.",
            "user_reply": "Datei erstellt: `src/main.py`.",
            "internal_summary": "Executed actions: create_file. Blocked actions: none. Approval request created: no. Self-check: Inspected files after execution: src/main.py. Follow-up: No runtime validation has been requested yet for the changed Python files.",
            "actions_executed": [{"action_type": "create_file", "target": "src/main.py"}],
            "actions_blocked": [],
            "workspace": str(tmp_path / "workspaces" / "thread"),
            "internal_payload": {
                "language": "en",
                "thread_id": "ignored",
                "task": "Create one starter file.",
                "self_check": {
                    "inspected_files": [{"path": "src/main.py", "preview": "print('ok')\n", "bytes": 12}],
                    "touched_paths": ["src/main.py"],
                    "follow_up": ["No runtime validation has been requested yet for the changed Python files."],
                    "summary": "Inspected files after execution: src/main.py. Follow-up: No runtime validation has been requested yet for the changed Python files.",
                },
            },
        }
    )
    review_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Review reply",
            "user_reply": "Review reply",
            "internal_summary": "The latest coding step should be validated with one focused runtime check.",
        }
    )
    workflow.delegation.coding_agent = coding_worker
    workflow.delegation.reviewer_agent = review_worker

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte erstelle eine kleine Python-Datei src/main.py.",
    )

    assert result["route"] == "coding"
    assert len(review_worker.calls) == 1
    assert "Self-check summary:" in review_worker.calls[0]["internal_task"]
    assert "File: src/main.py" in review_worker.calls[0]["internal_task"]
    assert "Preview: print('ok')" in review_worker.calls[0]["internal_task"]
    artifacts = {artifact["kind"]: artifact for artifact in store.list_artifacts(result["thread"]["id"])}
    assert artifacts["coding_status"]["content"]["self_check"]["inspected_files"][0]["path"] == "src/main.py"
    assert artifacts["implementation_review"]["summary"]


def test_manager_skips_post_coding_review_for_empty_bootstrap_file(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.schemas.coding_actions import CodingActionBatch, CreateFileAction
    from ai_hub.schemas.coding_delegation import CodingDelegationPlan
    from ai_hub.schemas.manager_plan import ManagerPlan

    class EmptyBootstrapPlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Create one placeholder file.",
                    decision="coding",
                    reason="Needs a tiny bootstrap step.",
                    user_reply="Ich delegiere das Coding.",
                    internal_task_for_worker="Create one starter file.",
                    approval_needed=False,
                    coding_plan=CodingDelegationPlan(
                        summary="Create one empty file.",
                        rationale="Bootstrap only.",
                        approval_needed=False,
                        actions=CodingActionBatch(
                            actions=[CreateFileAction(path="src/main.py", content="", content_inferred=False)]
                        ),
                    ),
                ),
            }

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=EmptyBootstrapPlanner())
    coding_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Datei erstellt: `src/main.py`.",
            "user_reply": "Datei erstellt: `src/main.py`.",
            "internal_summary": "Executed actions: create_file.",
            "actions_executed": [{"action_type": "create_file", "target": "src/main.py"}],
            "actions_blocked": [],
            "workspace": str(tmp_path / "workspaces" / "thread"),
            "internal_payload": {
                "language": "en",
                "thread_id": "ignored",
                "task": "Create one starter file.",
                "self_check": {
                    "inspected_files": [{"path": "src/main.py", "preview": "", "bytes": 0}],
                    "touched_paths": ["src/main.py"],
                    "follow_up": [],
                    "summary": "Inspected files after execution: src/main.py.",
                },
            },
        }
    )
    review_worker = RecordingWorker(
        {
            "status": "completed",
            "reply": "Should not be used",
            "user_reply": "Should not be used",
            "internal_summary": "Should not be used",
        }
    )
    workflow.delegation.coding_agent = coding_worker
    workflow.delegation.reviewer_agent = review_worker

    result = workflow.handle_chat(thread_id=None, user_message="Bitte lege src/main.py an.")

    assert result["route"] == "coding"
    assert len(review_worker.calls) == 0


def test_reset_thread_clears_thread_artifacts(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore

    store = HubStore()
    thread = store.ensure_thread(None)
    store.upsert_artifact(
        thread_id=thread["id"],
        kind="project_brief",
        title="Project Brief",
        summary="To be cleared.",
        content={"state": "stale"},
    )

    store.reset_thread(thread["id"])

    assert store.list_artifacts(thread["id"]) == []


def test_relative_paths_with_slashes_are_not_falsely_blocked(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.tools.file_tools import read_file

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

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
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte lösche den Ordner temp_only wieder.",
    )

    assert result["route"] == "coding"


def test_manager_planner_can_be_prepared_without_enabling_live_control(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.llm.manager_planner import ManagerPlanner

    planner = ManagerPlanner(enabled=False)

    try:
        planner.plan(history_text="Noch kein Verlauf.", user_message="Bitte plane den nächsten Schritt.", user_language="de")
    except RuntimeError as exc:
        assert "disabled" in str(exc).lower()
    else:
        raise AssertionError("Expected planner.plan() to raise when disabled.")


def test_manager_answers_capabilities_question_compactly(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

    result = workflow.handle_chat(thread_id=None, user_message="Was kannst du aktuell in diesem System tun?")

    assert result["route"] == "direct"
    assert result["reply"].startswith("LLM-Antwort:")


def test_manager_routes_review_requests_to_reviewer_agent(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.reviewer_agent import ReviewerAgent
    from ai_hub.schemas.manager_plan import ManagerPlan
    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    class FakeReviewerClient:
        def generate(self, model: str, prompt: str, temperature: float = 0.2) -> str:
            return """
            {
              "summary": "A focused review was produced.",
              "findings": ["There is no explicit regression check."],
              "assumptions": ["The latest change touches the agent flow."],
              "open_questions": ["Which scenario matters most?"],
              "recommendation": "Run one focused regression test next.",
              "user_reply": "Ich habe den Sachverhalt kritisch geprüft."
            }
            """

    store = HubStore()
    class FakePlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Review task.",
                    decision="review",
                    reason="Needs a critical second opinion.",
                    user_reply="Ich delegiere die Prüfung.",
                    internal_task_for_worker="Review the latest plan critically and identify risks.",
                    approval_needed=False,
                ),
            }

    workflow = ManagerWorkflow(store=store, planner=FakePlanner())
    workflow.delegation.reviewer_agent = ReviewerAgent(client=FakeReviewerClient())

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte prüfe den letzten Plan kritisch und gib mir eine Zweitmeinung.",
    )

    assert result["route"] == "review"
    assert "Ich delegiere das an den Kritiker-Agenten." in result["reply"]
    assert "Kritische Punkte:" in result["reply"]


def test_manager_uses_llm_plan_when_available(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    class FakePlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            from ai_hub.agents.coding_agent import CodingAgent
            from ai_hub.schemas.coding_delegation import CodingDelegationPlan
            from ai_hub.schemas.manager_plan import ManagerPlan

            batch = CodingAgent().build_action_batch_from_text(user_message)

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

    assert result["route"] == "error"
    assert "Manager-Fehler:" in result["reply"]
    assert "implausible direct reply" in result["reply"].lower()
    assert result["messages"][-1]["meta"]["manager_source"] == "error"


def test_manager_reports_planner_error_in_chat(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    class BrokenPlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            raise RuntimeError("Ollama offline")

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=BrokenPlanner())

    result = workflow.handle_chat(thread_id=None, user_message="Was kannst du aktuell in diesem System tun?")

    assert result["route"] == "error"
    assert "Manager-Fehler:" in result["reply"]
    assert "Ollama offline" in result["reply"]
    assert result["messages"][-1]["meta"]["manager_source"] == "error"


def test_manager_reports_missing_plan_in_chat(monkeypatch, tmp_path):
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

    assert result["route"] == "error"
    assert "Manager-Fehler:" in result["reply"]
    assert "did not return a valid plan" in result["reply"]


def test_llm_coding_plan_is_used_without_heuristic_override(monkeypatch, tmp_path):
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
                    summary="User asks for architecture advice.",
                    decision="coding",
                    reason="Coding might help later.",
                    user_reply="Ich würde das in Manager, Coding-Agent und Research-Agent aufteilen.",
                    internal_task_for_worker="Create a starter workspace structure.",
                    approval_needed=False,
                    coding_plan=CodingDelegationPlan(
                        summary="Create a starter note.",
                        rationale="The planner explicitly wants a workspace artifact.",
                        actions=CodingActionBatch(
                            actions=[CreateFileAction(path="starter/plan.md", content="# Plan\n", content_inferred=False)]
                        ),
                    ),
                ),
            }

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=FakePlanner())

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Ich möchte mit dir ein kleines Python-Projekt starten, das später über Tailscale und Web Push erreichbar ist. Wie würdest du das strukturieren?",
    )

    assert result["route"] == "coding"
    assert "Datei erstellt:" in result["reply"]
    assert result["messages"][-1]["meta"]["llm_decision"] == "coding"
    assert result["messages"][-1]["meta"]["final_decision"] == "coding"


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
                    coding_plan=CodingDelegationPlan(
                        summary="Structured coding plan",
                        rationale="Deterministic planner provides structured actions for the test.",
                        actions=batch,
                    ),
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
    workflow = ManagerWorkflow(store=store, planner=DeterministicPlanner())

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte erstelle in deinem Workspace eine hallo.txt-Datei mit dem Inhalt Hallo",
    )

    assistant_message = result["messages"][-1]
    internal_payload = assistant_message["meta"]["internal_payload"]
    assert internal_payload["language"] == "en"
    assert internal_payload["task"]
    assert "Workspace" not in internal_payload["task"]


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

    agent = CodingAgent(
        client=FakeCodingClient(
            """
            {
              "actions": [
                {"action_type": "make_directory", "path": "demo"},
                {"action_type": "create_file", "path": "demo/notes.txt", "content": "", "content_inferred": false}
              ]
            }
            """
        )
    )
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

    agent = CodingAgent(
        client=FakeCodingClient(
            """
            {
              "actions": [
                {"action_type": "make_directory", "path": "src"},
                {"action_type": "create_file", "path": "src/main.py", "content": "print(\\"Hello World\\")\\n", "content_inferred": false},
                {"action_type": "request_execution", "target": "src/main.py"}
              ]
            }
            """
        )
    )
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

    agent = CodingAgent(
        client=FakeCodingClient(
            """
            {
              "actions": [
                {"action_type": "make_directory", "path": "demo"},
                {"action_type": "create_file", "path": "demo/structured.txt", "content": "IGNORED", "content_inferred": false}
              ]
            }
            """
        )
    )
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
