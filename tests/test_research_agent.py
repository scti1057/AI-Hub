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


def test_approval_execution_runs_after_approval(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.tools.file_tools import write_file

    store = HubStore()
    workflow = ManagerWorkflow(store=store)

    thread = store.ensure_thread(None, "Sandbox Test")
    write_file(thread["id"], "main.py", "print('sandbox-ok')\n")

    result = workflow.handle_chat(
        thread_id=thread["id"],
        user_message="Bitte starte main.py im Python-Workspace.",
    )

    approval = workflow.approve_execution(result["approval_request"]["id"])

    assert approval["status"] == "executed"
    assert approval["result"]["returncode"] == 0
    assert "sandbox-ok" in approval["result"]["stdout"]
    assert approval["result"]["sandbox_backend"] in {"bwrap", "subprocess_fallback", "subprocess"}
