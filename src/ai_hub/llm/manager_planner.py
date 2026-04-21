import hashlib
import json
import logging
import re
from pathlib import Path

from ai_hub.config import (
    MANAGER_DEBUG_LOG_MAX_CHARS,
    MANAGER_DEBUG_LOG_PROMPTS,
    MANAGER_LLM_PLANNING_ENABLED,
    MANAGER_OLLAMA_ENABLED,
)
from ai_hub.logging_config import log_event, log_text_block, setup_logging
from ai_hub.llm.model_router import ModelRouter
from ai_hub.llm.ollama_client import OllamaClient
from ai_hub.schemas.manager_plan import ManagerPlan


logger = logging.getLogger(__name__)
setup_logging()


class ManagerPlanner:
    def __init__(
        self,
        enabled: bool = MANAGER_LLM_PLANNING_ENABLED,
        ollama_enabled: bool = MANAGER_OLLAMA_ENABLED,
        client: OllamaClient | None = None,
    ) -> None:
        self.enabled = enabled
        self.ollama_enabled = ollama_enabled
        self.client = client or OllamaClient()
        self.model = ModelRouter.get_model_for_role("manager")
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "manager.txt"
        return prompt_path.read_text(encoding="utf-8").strip()

    def plan(
        self,
        history_text: str,
        user_message: str,
        user_language: str = "de",
    ) -> dict | None:
        if not self.enabled or not self.ollama_enabled:
            raise RuntimeError("Manager planner is disabled.")

        prompt = f"""
{self.system_prompt}

You are assisting a local manager agent inside a secure orchestration backend.
You may help understand the request, choose the route, and draft the user-facing reply.
You must never bypass approval rules, workspace restrictions, or backend tool policies.
Internal worker instructions must be written in English.
All stored repo plans, step names, validation notes, and completion criteria must be written in English.
The user-facing reply must be written in this language code: {user_language}.
When you choose decision="coding" for a bounded implementation step, prefer meaningful file contents over empty stubs.
Only use empty file contents when the file is intentionally a placeholder such as an empty __init__.py.
If the user asked for a project structure or a concrete first step, the coding_plan should create a coherent small implementation, not only comments or placeholder headings.
Use the word "steps", not "slices", when you describe implementation sequencing.
Treat explicit user prohibitions and confirmed implementation decisions from the current message or project memory as hard constraints.
Do not propose dependency installs, pytest runs, execution requests, or automatic follow-on repairs when those constraints explicitly forbid them.
Do not draft user-facing replies that dump raw file contents or large code blocks unless the user explicitly asked to inspect file contents.

Recent thread context:
{history_text}

The recent thread context may include a "Project memory" section summarizing stored artifacts.
Treat that section as grounded project state and prefer it over re-inventing context from scratch.
When project memory mentions review findings, repair tasks, change snapshots, or current phase, keep your route and plan aligned with that state.

New user message:
{user_message}

Return JSON only with these fields:
{{
  "summary": "short summary",
  "decision": "direct|plan|coding|research|review",
  "reason": "why this route",
  "user_reply": "short user-facing reply in the user's language",
  "internal_task_for_worker": "English instruction for the worker or empty string",
  "approval_needed": true_or_false,
  "coding_plan": {{
    "summary": "optional coding summary",
    "rationale": "optional coding rationale",
    "approval_needed": true_or_false,
    "actions": {{
      "actions": [
        {{"action_type": "make_directory", "path": "demo"}},
        {{"action_type": "create_file", "path": "demo/notes.txt", "content": "", "content_inferred": false}},
        {{"action_type": "read_file", "path": "demo/notes.txt"}},
        {{"action_type": "delete_path", "path": "demo/notes.txt"}},
        {{"action_type": "request_execution", "target": "src/main.py"}}
      ]
    }}
  }},
  "project_outline": {{
    "summary": "optional English project summary",
    "repo_structure": [
      "src/main.py - entry point",
      "src/game_logic.py - core rules"
    ],
    "steps": [
      "Create the repository scaffold and shared module boundaries.",
      "Implement the core game loop and wire the entry point."
    ],
    "validation_steps": [
      "Run the main entry point once for a smoke test."
    ],
    "completion_criteria": [
      "The project starts from main.py without import errors.",
      "The requested first scenario is ready to test."
    ],
    "autonomous_execution": true_or_false
  }}
}}

Use decision="plan" when the user wants strategy, architecture, milestones, implementation steps, or feedback before coding starts.
Only include a non-empty coding_plan when decision="coding". Otherwise set coding_plan to null.
Include project_outline when the request is a multi-file project, repo architecture task, or autonomous multi-step implementation. Otherwise set project_outline to null.
""".strip()

        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        log_event(
            logger,
            "manager_ollama_request",
            model=self.model,
            prompt_chars=len(prompt),
            prompt_digest=prompt_digest,
            history_chars=len(history_text),
            user_message_chars=len(user_message),
            user_language=user_language,
        )
        if MANAGER_DEBUG_LOG_PROMPTS:
            log_text_block(
                logger,
                "manager_prompt_body",
                prompt,
                max_chars=MANAGER_DEBUG_LOG_MAX_CHARS,
                model=self.model,
                prompt_digest=prompt_digest,
            )
        response = self.client.generate(model=self.model, prompt=prompt)
        if MANAGER_DEBUG_LOG_PROMPTS:
            log_text_block(
                logger,
                "manager_raw_response",
                response,
                max_chars=MANAGER_DEBUG_LOG_MAX_CHARS,
                model=self.model,
                prompt_digest=prompt_digest,
            )
        plan = self._parse_plan(response)
        log_event(
            logger,
            "manager_ollama_response",
            decision=plan.decision,
            prompt_digest=prompt_digest,
            response_chars=len(response),
            summary_chars=len(plan.summary),
        )
        return {
            "enabled": True,
            "source": "ollama",
            "plan": plan,
            "raw_response": response,
        }

    def _parse_plan(self, response: str) -> ManagerPlan:
        json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not json_match:
            raise ValueError("Manager planner did not return JSON.")
        payload = json.loads(json_match.group(0))
        return ManagerPlan.model_validate(payload)
