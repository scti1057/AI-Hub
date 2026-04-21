import importlib

from ai_hub.tools.web_search import WebSearchClient


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


class FakeResearchClient:
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


class FakeResponse:
    def __init__(self, payload: dict, ok: bool = True, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get(self, url: str, **kwargs):
        self.calls.append({"method": "get", "url": url, "kwargs": kwargs})
        if "api.search.brave.com" in url:
            return FakeResponse(
                {
                    "web": {
                        "results": [
                            {
                                "title": "Docs",
                                "url": "https://example.com/docs",
                                "description": "Official documentation snippet.",
                            }
                        ]
                    }
                }
            )
        return FakeResponse({}, text="<html><body>Fetched page preview text.</body></html>")


class FakeStore:
    def __init__(self, artifacts: dict[str, dict]) -> None:
        self.artifacts = artifacts

    def list_artifacts(self, thread_id: str) -> list[dict]:
        return list(self.artifacts.values())

    def get_artifact(self, thread_id: str, kind: str) -> dict | None:
        return self.artifacts.get(kind)


def test_research_agent_uses_research_model_and_parses_llm_reply(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.research_agent import ResearchAgent

    client = FakeResearchClient(
        """
        {
          "summary": "The request needs a concise comparison.",
          "findings": ["Option A is simpler.", "Option B is more flexible."],
          "assumptions": ["No latency budget was provided."],
          "open_questions": ["Is local-only operation mandatory?"],
          "recommendation": "Start with Option A and validate it with one realistic task.",
          "user_reply": "Ich habe die Anfrage als Vergleich strukturiert und die wichtigsten Spannungen herausgearbeitet."
        }
        """
    )
    agent = ResearchAgent(client=client)

    result = agent.handle_task(
        thread_id="thread-r1",
        user_task="Bitte vergleiche zwei Ansätze für die Recherche-Pipeline.",
        history=[],
    )

    assert result["status"] == "completed"
    assert client.calls[0]["model"] == "qwen3:30b"
    assert "Kernpunkte:" in result["reply"]
    assert "Annahmen:" in result["reply"]
    assert "Offene Fragen:" in result["reply"]
    assert result["internal_payload"]["source"] == "ollama"
    assert result["internal_payload"]["task"].startswith("research:")


def test_research_agent_falls_back_when_llm_fails(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.research_agent import ResearchAgent

    client = FakeResearchClient("{}", fail=True)
    agent = ResearchAgent(client=client)

    result = agent.handle_task(
        thread_id="thread-r2",
        user_task="Bitte mache eine kurze Rechercheeinschätzung.",
        history=[{"role": "user", "content": "Vorheriger Kontext"}],
    )

    assert result["status"] == "error"
    assert "Research-Agent Fehler:" in result["reply"]
    assert result["internal_payload"]["source"] == "error"
    assert result["internal_payload"]["error"]["message"]


def test_web_search_client_parses_brave_results_and_fetches_page_preview():
    session = FakeSession()
    client = WebSearchClient(
        provider="brave",
        brave_api_key="test-key",
        session=session,
        fetch_pages=True,
        fetch_top_n=1,
    )

    result = client.search("official docs")

    assert result["provider"] == "brave"
    assert result["results"][0]["url"] == "https://example.com/docs"
    assert "Fetched page preview text." in result["results"][0]["page_preview"]


def test_research_agent_uses_web_search_context_when_available(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.research_agent import ResearchAgent

    client = FakeResearchClient(
        """
        {
          "summary": "Used documentation context.",
          "findings": ["The docs mention a search endpoint."],
          "assumptions": [],
          "open_questions": [],
          "recommendation": "Use the official search provider abstraction.",
          "user_reply": "Ich habe die Web-Recherche mit Quellenkontext ausgewertet."
        }
        """
    )
    session = FakeSession()
    web_search = WebSearchClient(
        provider="brave",
        brave_api_key="test-key",
        session=session,
        fetch_pages=True,
        fetch_top_n=1,
    )
    agent = ResearchAgent(client=client, web_search=web_search)

    result = agent.handle_task(
        thread_id="thread-r3",
        user_task="Bitte recherchiere die aktuelle offizielle Dokumentation fuer Web Search APIs.",
        history=[],
    )

    assert result["status"] == "completed"
    assert "Web research context:" in client.calls[0]["prompt"]
    assert "Official documentation snippet." in client.calls[0]["prompt"]
    assert result["internal_payload"]["sources"][0]["url"] == "https://example.com/docs"


def test_research_agent_uses_stored_artifact_memory(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.research_agent import ResearchAgent

    client = FakeResearchClient(
        """
        {
          "summary": "The stored project memory narrows the analysis.",
          "findings": ["The last review still expects a repair on the route files."],
          "assumptions": [],
          "open_questions": ["Should the next step stay focused on the route repair?"],
          "recommendation": "Keep the next implementation bounded to the known repair scope.",
          "user_reply": "Ich habe die gespeicherte Projektmemory in die Analyse einbezogen."
        }
        """
    )
    store = FakeStore(
        {
            "project_state": {
                "kind": "project_state",
                "summary": "Project is in needs_repair phase.",
                "content": {"phase": "needs_repair"},
            },
            "manager_constraints": {
                "kind": "manager_constraints",
                "summary": "Do not install dependencies or run pytest yet.",
                "content": {
                    "confirmed_directives": [
                        "Use automatic integer task IDs starting at 1.",
                        "Store the JSON file at the project root as tasks.json.",
                    ],
                    "forbid_dependency_install": True,
                    "forbid_pytest": True,
                },
            },
            "coding_change_snapshot": {
                "kind": "coding_change_snapshot",
                "summary": "1 modified",
                "content": {
                    "step_goal": "Repair the route files.",
                    "summary": "1 modified",
                    "review_verdict": "needs_repair",
                    "snapshot_changes": [{"path": "src/app/page.tsx", "status": "modified"}],
                },
            },
            "implementation_review": {
                "kind": "implementation_review",
                "summary": "The route still needs one focused repair.",
                "content": {
                    "summary": "The route still needs one focused repair.",
                    "internal_payload": {
                        "verdict": "needs_repair",
                        "project_status": "in_progress",
                        "repair_tasks": ["Fix the route variant mismatch."],
                    },
                },
            },
        }
    )
    agent = ResearchAgent(client=client, store=store)

    result = agent.handle_task(
        thread_id="thread-r-memory",
        user_task="Bitte analysiere den naechsten sicheren Schritt.",
        history=[],
    )

    assert result["status"] == "completed"
    assert "Stored project/artifact memory:" in client.calls[0]["prompt"]
    assert "Confirmed user constraints and decisions:" in client.calls[0]["prompt"]
    assert "Use automatic integer task IDs starting at 1." in client.calls[0]["prompt"]
    assert "Project is in needs_repair phase." in client.calls[0]["prompt"]
    assert "Fix the route variant mismatch." in client.calls[0]["prompt"]
    assert result["internal_payload"]["artifact_context"]["latest_change_snapshot"]["step_goal"] == "Repair the route files."


def test_manager_routes_research_requests_to_research_agent(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.agents.research_agent import ResearchAgent
    from ai_hub.schemas.manager_plan import ManagerPlan
    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow

    client = FakeResearchClient(
        """
        {
          "summary": "Structured research handoff.",
          "findings": ["The request is analytical, not executable."],
          "assumptions": ["No code changes are required yet."],
          "open_questions": ["Which artifact should be reviewed first?"],
          "recommendation": "Start with the README and existing schemas.",
          "user_reply": "Ich habe die Anfrage als Analyseauftrag eingeordnet."
        }
        """
    )

    store = HubStore()
    class FakePlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Route to research.",
                    decision="research",
                    reason="Needs research analysis.",
                    user_reply="Ich delegiere die Analyse.",
                    internal_task_for_worker="Analyze the existing architecture and summarize open issues.",
                    approval_needed=False,
                    coding_plan=None,
                ),
            }

    workflow = ManagerWorkflow(store=store, planner=FakePlanner())
    workflow.delegation.research_agent = ResearchAgent(client=client)

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte mache eine Recherche zur bestehenden Architektur und fasse die offenen Punkte zusammen.",
    )

    assert result["route"] == "research"
    assert "Ich delegiere das an den Research-Agenten." in result["reply"]
    assert "Kernpunkte:" in result["reply"]


def test_manager_stores_web_research_notes_when_sources_exist(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)

    from ai_hub.memory.store import HubStore
    from ai_hub.orchestration.workflow import ManagerWorkflow
    from ai_hub.schemas.manager_plan import ManagerPlan

    class FakePlanner:
        def plan(self, history_text: str, user_message: str, user_language: str):
            return {
                "enabled": True,
                "source": "ollama",
                "plan": ManagerPlan(
                    summary="Route to research.",
                    decision="research",
                    reason="Needs research analysis.",
                    user_reply="Ich delegiere die Analyse.",
                    internal_task_for_worker="Analyze the external documentation and summarize it.",
                    approval_needed=False,
                    coding_plan=None,
                ),
            }

    class FakeResearchWorker:
        def handle_task(self, thread_id, user_task, history, internal_task=None):
            return {
                "status": "completed",
                "reply": "Ich habe recherchiert.",
                "user_reply": "Ich habe recherchiert.",
                "tool_results": [],
                "internal_summary": "Used web-backed research.",
                "internal_payload": {
                    "language": "en",
                    "thread_id": thread_id,
                    "task": f"research: {internal_task}",
                    "source": "ollama",
                    "summary": "Used web-backed research.",
                    "open_questions": ["Should we cache the result?"],
                    "sources": [
                        {
                            "title": "Official Docs",
                            "url": "https://example.com/docs",
                            "source": "brave",
                        }
                    ],
                },
            }

    store = HubStore()
    workflow = ManagerWorkflow(store=store, planner=FakePlanner())
    workflow.delegation.research_agent = FakeResearchWorker()

    result = workflow.handle_chat(
        thread_id=None,
        user_message="Bitte recherchiere die offizielle Dokumentation zur Websuche.",
    )

    artifacts = {artifact["kind"]: artifact for artifact in store.list_artifacts(result["thread"]["id"])}
    assert "research_notes" in artifacts
    assert "web_research_notes" in artifacts
    assert artifacts["web_research_notes"]["content"]["sources"][0]["url"] == "https://example.com/docs"
