import importlib


def configure_workspace(monkeypatch, tmp_path):
    workspace_root = tmp_path / "workspaces"

    import ai_hub.config as config
    import ai_hub.tools.file_tools as file_tools

    monkeypatch.setattr(config, "CODING_WORKSPACE_ROOT", workspace_root)
    monkeypatch.setattr(file_tools, "CODING_WORKSPACE_ROOT", workspace_root)

    importlib.reload(file_tools)
    return file_tools


class FakeExplorerClient:
    def __init__(self, response: str, fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.calls: list[dict] = []

    def generate(self, model: str, prompt: str, temperature: float = 0.1) -> str:
        self.calls.append({"model": model, "prompt": prompt, "temperature": temperature})
        if self.fail:
            raise RuntimeError("ollama unavailable")
        return self.response


class FakeStore:
    def __init__(self, artifacts: dict[str, dict]) -> None:
        self.artifacts = artifacts

    def list_artifacts(self, thread_id: str) -> list[dict]:
        return list(self.artifacts.values())

    def get_artifact(self, thread_id: str, kind: str) -> dict | None:
        return self.artifacts.get(kind)


def test_explorer_agent_grounds_context_in_existing_files(monkeypatch, tmp_path):
    file_tools = configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.explorer_agent import ExplorerAgent

    file_tools.write_file("thread-explorer", "src/app/page.tsx", "export default function Page() { return <main />; }\n")
    file_tools.write_file("thread-explorer", "src/app/page.js", "export default function LegacyPage() { return null; }\n")
    file_tools.write_file("thread-explorer", "src/app/layout.tsx", "export default function Layout({ children }) { return children; }\n")
    file_tools.write_file("thread-explorer", "src/components/card.tsx", "export function Card() { return <section />; }\n")

    client = FakeExplorerClient(
        """
        {
          "summary": "The next step touches the app router entrypoints.",
          "repo_overview": "Next.js app router files live in src/app with a small components directory.",
          "relevant_paths": [],
          "suggested_definition_of_done": ["Change only the app router files that are already present."],
          "suggested_checks": ["Keep route file variants coherent."],
          "risks": ["There are duplicate page route variants."],
          "user_reply": "Ich habe die relevante Struktur fuer den naechsten Schritt eingegrenzt."
        }
        """
    )

    agent = ExplorerAgent(client=client)
    result = agent.handle_task(
        thread_id="thread-explorer",
        user_task="Bitte passe die page route an und halte die Struktur konsistent.",
        history=[],
        internal_task="Inspect the route files for the next coding step.",
    )

    assert result["status"] == "completed"
    assert client.calls[0]["model"] == "devstral:24b"
    payload = result["internal_payload"]
    assert "src/app/page.tsx" in payload["candidate_paths"]
    assert "src/app/page.js" in payload["candidate_paths"]
    assert "src/app/page.tsx" in payload["relevant_paths"]
    assert payload["snapshots"]
    assert any("App Router structure is present" in item for item in payload["structure_signals"])
    assert any("Multiple route file variants exist" in item for item in payload["structure_signals"])
    assert "Candidate file previews:" in client.calls[0]["prompt"]


def test_explorer_agent_falls_back_cleanly(monkeypatch, tmp_path):
    configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.explorer_agent import ExplorerAgent

    client = FakeExplorerClient("{}", fail=True)
    agent = ExplorerAgent(client=client)

    result = agent.handle_task(
        thread_id="thread-explorer-error",
        user_task="Bitte schaue dir das Repo an.",
        history=[],
    )

    assert result["status"] == "error"
    assert "Explorer-Agent Fehler:" in result["reply"]
    assert result["internal_payload"]["source"] == "error"


def test_explorer_agent_uses_stored_change_snapshot_memory(monkeypatch, tmp_path):
    file_tools = configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.explorer_agent import ExplorerAgent

    file_tools.write_file("thread-explorer-memory", "src/app/page.tsx", "export default function Page() { return <main />; }\n")
    store = FakeStore(
        {
            "coding_change_snapshot": {
                "kind": "coding_change_snapshot",
                "summary": "1 modified",
                "content": {
                    "step_goal": "Update the main route.",
                    "summary": "1 modified",
                    "review_verdict": "needs_repair",
                    "snapshot_changes": [{"path": "src/app/page.tsx", "status": "modified"}],
                },
            },
            "implementation_review": {
                "kind": "implementation_review",
                "summary": "The route still needs one follow-up fix.",
                "content": {
                    "summary": "The route still needs one follow-up fix.",
                    "internal_payload": {
                        "verdict": "needs_repair",
                        "project_status": "in_progress",
                        "repair_tasks": ["Fix the route variant mismatch."],
                    },
                },
            },
        }
    )

    client = FakeExplorerClient(
        """
        {
          "summary": "The next step should build on the stored change history.",
          "repo_overview": "The app route file remains the key integration point.",
          "relevant_paths": ["src/app/page.tsx"],
          "suggested_definition_of_done": ["Resolve the stored route mismatch without broad rewrites."],
          "suggested_checks": ["Confirm the route file variants are coherent."],
          "risks": ["The last review already flagged a remaining route issue."],
          "user_reply": "Ich habe den letzten Aenderungskontext in die Repo-Analyse einbezogen."
        }
        """
    )

    agent = ExplorerAgent(client=client, store=store)
    result = agent.handle_task(
        thread_id="thread-explorer-memory",
        user_task="Bitte analysiere den naechsten Reparatur-Schritt fuer die Route.",
        history=[],
    )

    assert result["status"] == "completed"
    assert "Latest stored change snapshot:" in client.calls[0]["prompt"]
    assert "src/app/page.tsx: modified" in client.calls[0]["prompt"]
    assert "Fix the route variant mismatch." in client.calls[0]["prompt"]
    assert result["internal_payload"]["artifact_context"]["latest_change_snapshot"]["step_goal"] == "Update the main route."


def test_explorer_agent_surfaces_validation_paths_from_project_memory(monkeypatch, tmp_path):
    file_tools = configure_workspace(monkeypatch, tmp_path)

    from ai_hub.agents.explorer_agent import ExplorerAgent

    file_tools.write_file("thread-explorer-validation", "src/main.py", "print('ready')\n")
    file_tools.write_file("thread-explorer-validation", "tests/test_main.py", "def test_ready():\n    assert True\n")
    store = FakeStore(
        {
            "project_plan": {
                "kind": "project_plan",
                "summary": "Plan with validation.",
                "content": {
                    "summary": "Plan with validation.",
                    "validation_steps": [
                        "Run one smoke test for src/main.py.",
                        "Run pytest tests/test_main.py -v.",
                    ],
                },
            },
            "project_state": {
                "kind": "project_state",
                "summary": "Validation still missing.",
                "content": {
                    "phase": "implementation",
                    "validation_status": "not_requested",
                    "validation_summary": "Validation is still missing for the latest implementation step.",
                },
            },
        }
    )

    client = FakeExplorerClient(
        """
        {
          "summary": "The next step should keep the runtime and test entrypoints in view.",
          "repo_overview": "The repo has a small runtime entrypoint and a focused pytest file.",
          "relevant_paths": ["src/main.py"],
          "validation_relevant_paths": [],
          "suggested_definition_of_done": ["Keep the entrypoint aligned with the focused test path."],
          "suggested_checks": ["Use the stored validation flow as the next check."],
          "risks": ["Validation is still missing for the latest implementation step."],
          "user_reply": "Ich habe die relevanten Validierungspfade fuer den naechsten Schritt markiert."
        }
        """
    )

    agent = ExplorerAgent(client=client, store=store)
    result = agent.handle_task(
        thread_id="thread-explorer-validation",
        user_task="Bitte bereite den nächsten Schritt für src/main.py vor.",
        history=[],
    )

    assert result["status"] == "completed"
    payload = result["internal_payload"]
    assert "src/main.py" in payload["validation_relevant_paths"]
    assert "tests/test_main.py" in payload["validation_relevant_paths"]
    assert "project_state.validation_status: not_requested" in client.calls[0]["prompt"]
    assert "project_plan.validation_step: Run pytest tests/test_main.py -v." in client.calls[0]["prompt"]
