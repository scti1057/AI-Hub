import hashlib
import json
import logging
import re
from pathlib import Path

from ai_hub.config import RESEARCH_DEBUG_LOG_MAX_CHARS, RESEARCH_DEBUG_LOG_PROMPTS
from ai_hub.language_policy import LanguagePolicy
from ai_hub.logging_config import log_event, log_text_block, setup_logging
from ai_hub.llm.model_router import ModelRouter
from ai_hub.llm.ollama_client import LLMServiceError, OllamaClient
from ai_hub.memory.history import format_thread_history
from ai_hub.memory.store import HubStore
from ai_hub.tools.web_search import WebSearchClient, format_search_context


logger = logging.getLogger(__name__)
setup_logging()


class ResearchAgent:
    role = "research"

    def __init__(
        self,
        language_policy: LanguagePolicy | None = None,
        client: OllamaClient | None = None,
        web_search: WebSearchClient | None = None,
        store: HubStore | None = None,
    ) -> None:
        self.language_policy = language_policy or LanguagePolicy()
        self.client = client or OllamaClient()
        self.web_search = web_search or WebSearchClient()
        self.store = store
        self.model = ModelRouter.get_model_for_role("research")
        self.system_prompt = self._load_system_prompt()

    def handle_task(
        self,
        thread_id: str,
        user_task: str,
        history: list[dict],
        internal_task: str | None = None,
    ) -> dict:
        language_context = self.language_policy.build_context(user_task)
        summary = format_thread_history(history, limit=4)
        worker_task = internal_task or language_context.internal_message
        web_context = "No web search used."
        search_payload = None
        artifact_context = self._load_artifact_context(thread_id)

        try:
            if self._should_use_web_search(user_task, worker_task):
                search_query = self._build_search_query(user_task, worker_task)
                search_payload = self.web_search.search(search_query)
                web_context = format_search_context(search_payload)
            prompt = self._build_prompt(
                history_summary=summary,
                user_task=user_task,
                internal_task=worker_task,
                user_language=language_context.user_language,
                web_context=web_context,
                artifact_memory=self._artifact_memory_block(artifact_context),
                latest_constraints_memory=self._latest_constraints_block(artifact_context),
                latest_change_memory=self._latest_change_block(artifact_context),
                latest_review_memory=self._latest_review_block(artifact_context),
            )
            prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            log_event(
                logger,
                "research_ollama_request",
                thread_id=thread_id,
                model=self.model,
                history_chars=len(summary),
                worker_task_chars=len(worker_task),
                prompt_chars=len(prompt),
                prompt_digest=prompt_digest,
                worker_task_preview=worker_task[:160],
                used_web_search=bool(search_payload),
            )
            if RESEARCH_DEBUG_LOG_PROMPTS:
                log_text_block(
                    logger,
                    "research_prompt_body",
                    prompt,
                    max_chars=RESEARCH_DEBUG_LOG_MAX_CHARS,
                    thread_id=thread_id,
                    model=self.model,
                    prompt_digest=prompt_digest,
                )
            raw_response = self.client.generate(model=self.model, prompt=prompt, temperature=0.2)
            log_event(
                logger,
                "research_generate_completed",
                thread_id=thread_id,
                model=self.model,
                prompt_digest=prompt_digest,
                raw_response_chars=len(raw_response),
            )
            if RESEARCH_DEBUG_LOG_PROMPTS:
                log_text_block(
                    logger,
                    "research_raw_response",
                    raw_response,
                    max_chars=RESEARCH_DEBUG_LOG_MAX_CHARS,
                    thread_id=thread_id,
                    model=self.model,
                    prompt_digest=prompt_digest,
                )
            parsed = self._parse_response(raw_response)
            reply = self._render_reply(parsed, language_context.user_language)
            log_event(
                logger,
                "research_ollama_response",
                thread_id=thread_id,
                model=self.model,
                prompt_digest=prompt_digest,
                summary_chars=len(parsed["summary"]),
                findings_count=len(parsed["findings"]),
                open_questions_count=len(parsed["open_questions"]),
                reply_chars=len(reply),
            )
        except Exception as exc:
            log_event(
                logger,
                "research_error",
                thread_id=thread_id,
                model=self.model,
                error=str(exc),
            )
            return self._error_result(thread_id, worker_task, language_context.user_language, exc)

        return {
            "status": "completed",
            "reply": reply,
            "user_reply": reply,
            "tool_results": [],
            "internal_summary": parsed["summary"],
            "internal_payload": {
                "language": "en",
                "thread_id": thread_id,
                "task": f"research: {worker_task}",
                "policy": "Research is read-only and analytical. Use English for internal coordination.",
                "source": "ollama",
                "model": self.model,
                "summary": parsed["summary"],
                "findings": parsed["findings"],
                "assumptions": parsed["assumptions"],
                "open_questions": parsed["open_questions"],
                "recommendation": parsed["recommendation"],
                "sources": self._serialize_sources(search_payload),
                "artifact_context": artifact_context,
                "raw_response": raw_response,
            },
        }

    def _load_system_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "research_agent.txt"
        return prompt_path.read_text(encoding="utf-8").strip()

    def _build_prompt(
        self,
        history_summary: str,
        user_task: str,
        internal_task: str,
        user_language: str,
        web_context: str,
        artifact_memory: str,
        latest_constraints_memory: str,
        latest_change_memory: str,
        latest_review_memory: str,
    ) -> str:
        return f"""
{self.system_prompt}

You are working inside a local multi-agent backend.
You are the dedicated research worker. Stay analytical and read-only.
If web search context is provided below, you may use it and cite those sources conservatively.
Do not claim to have browsed the web, executed code, or changed files unless the prompt explicitly says so.
The user-facing text must be written in this language code: {user_language}.
Internal reasoning and worker coordination stay in English.

Recent thread context:
{history_summary}

Original user task:
{user_task}

Internal worker task:
{internal_task}

Stored project/artifact memory:
{artifact_memory}

Confirmed user constraints and decisions:
{latest_constraints_memory}

Latest stored change snapshot:
{latest_change_memory}

Latest stored implementation review:
{latest_review_memory}

Web research context:
{web_context}

Return JSON only with this shape:
{{
  "summary": "short analysis summary",
  "findings": ["fact or high-confidence observation"],
  "assumptions": ["assumption or uncertainty"],
  "open_questions": ["question to answer next"],
  "recommendation": "best next step in the user's language",
  "user_reply": "compact user-facing research answer in the user's language"
}}
""".strip()

    def _should_use_web_search(self, user_task: str, worker_task: str) -> bool:
        if not self.web_search.is_available():
            return False
        combined = f"{user_task}\n{worker_task}".lower()
        indicators = (
            "web",
            "internet",
            "latest",
            "aktuell",
            "heute",
            "today",
            "dokumentation",
            "documentation",
            "docs",
            "library",
            "framework",
            "api",
            "search",
            "recherche",
            "research",
            "compare",
            "vergleich",
        )
        return any(indicator in combined for indicator in indicators)

    def _build_search_query(self, user_task: str, worker_task: str) -> str:
        preferred = worker_task.strip() or user_task.strip()
        preferred = re.sub(r"\s+", " ", preferred).strip()
        return preferred[:240]

    def _serialize_sources(self, search_payload: dict | None) -> list[dict]:
        if not search_payload:
            return []
        serialized = []
        for item in search_payload.get("results") or []:
            serialized.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", ""),
                }
            )
        return serialized[:6]

    def _parse_response(self, response: str) -> dict:
        json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not json_match:
            raise ValueError("Research agent did not return JSON.")
        payload = json.loads(json_match.group(0))
        summary = str(payload.get("summary", "")).strip()
        user_reply = str(payload.get("user_reply", "")).strip()
        if not summary or not user_reply:
            raise ValueError("Research agent response is missing required fields.")
        return {
            "summary": summary,
            "findings": self._normalize_list(payload.get("findings")),
            "assumptions": self._normalize_list(payload.get("assumptions")),
            "open_questions": self._normalize_list(payload.get("open_questions")),
            "recommendation": str(payload.get("recommendation", "")).strip(),
            "user_reply": user_reply,
        }

    def _normalize_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized[:6]

    def _load_artifact_context(self, thread_id: str) -> dict:
        if self.store is None:
            return {
                "recent_artifacts": [],
                "manager_constraints": {},
                "latest_change_snapshot": {},
                "latest_review": {},
            }

        try:
            artifacts = self.store.list_artifacts(thread_id)
            latest_change = self.store.get_artifact(thread_id, "coding_change_snapshot")
            latest_review = self.store.get_artifact(thread_id, "implementation_review")
        except Exception:
            return {
                "recent_artifacts": [],
                "manager_constraints": {},
                "latest_change_snapshot": {},
                "latest_review": {},
            }

        preferred = (
            "project_state",
            "project_plan",
            "manager_constraints",
            "change_request_brief",
            "project_brief",
            "research_notes",
            "review_notes",
            "coding_step_contract",
            "coding_change_snapshot",
            "implementation_review",
        )
        artifact_map = {artifact["kind"]: artifact for artifact in artifacts}
        recent_artifacts: list[dict] = []
        for kind in preferred:
            artifact = artifact_map.get(kind)
            if artifact is None:
                continue
            recent_artifacts.append(
                {
                    "kind": artifact["kind"],
                    "summary": str(artifact.get("summary", "")).strip(),
                }
            )

        return {
            "recent_artifacts": recent_artifacts[:8],
            "manager_constraints": ((self.store.get_artifact(thread_id, "manager_constraints") or {}).get("content", {})),
            "latest_change_snapshot": (latest_change or {}).get("content", {}),
            "latest_review": (latest_review or {}).get("content", {}),
        }

    def _artifact_memory_block(self, artifact_context: dict) -> str:
        recent_artifacts = artifact_context.get("recent_artifacts") or []
        if not recent_artifacts:
            return "[no stored artifact memory available]"
        return "\n".join(
            f"- {item.get('kind', 'artifact')}: {item.get('summary', '') or '[no summary]'}"
            for item in recent_artifacts[:8]
        )

    def _latest_change_block(self, artifact_context: dict) -> str:
        latest_change = artifact_context.get("latest_change_snapshot") or {}
        if not latest_change:
            return "[no stored change snapshot available]"

        lines = []
        step_goal = str(latest_change.get("step_goal", "")).strip()
        if step_goal:
            lines.append(f"Step goal: {step_goal}")
        summary = str(latest_change.get("summary", "")).strip()
        if summary:
            lines.append(f"Summary: {summary}")
        review_verdict = str(latest_change.get("review_verdict", "")).strip()
        if review_verdict:
            lines.append(f"Review verdict: {review_verdict}")
        for change in (latest_change.get("snapshot_changes") or [])[:6]:
            path = str(change.get("path", "")).strip()
            status = str(change.get("status", "")).strip()
            if path and status:
                lines.append(f"- {path}: {status}")
        return "\n".join(lines) if lines else "[stored change snapshot is empty]"

    def _latest_constraints_block(self, artifact_context: dict) -> str:
        constraints = artifact_context.get("manager_constraints") or {}
        if not constraints:
            return "[no stored user constraints available]"
        lines = []
        for item in (constraints.get("confirmed_directives") or [])[:6]:
            lines.append(f"- {item}")
        if constraints.get("forbid_dependency_install"):
            lines.append("- Do not install dependencies unless the user explicitly changes that.")
        if constraints.get("forbid_pytest"):
            lines.append("- Do not run pytest unless the user explicitly changes that.")
        if constraints.get("forbid_validation_execution"):
            lines.append("- Do not request or run validation executions unless the user explicitly changes that.")
        if constraints.get("stop_after_step"):
            lines.append("- Stop after the current bounded step and report back before continuing.")
        return "\n".join(lines) if lines else "[stored user constraints are empty]"

    def _latest_review_block(self, artifact_context: dict) -> str:
        latest_review = artifact_context.get("latest_review") or {}
        if not latest_review:
            return "[no stored implementation review available]"

        lines = []
        summary = str(latest_review.get("summary", "")).strip()
        if summary:
            lines.append(f"Summary: {summary}")
        payload = latest_review.get("internal_payload") or {}
        verdict = str(payload.get("verdict", "")).strip()
        if verdict:
            lines.append(f"Verdict: {verdict}")
        project_status = str(payload.get("project_status", "")).strip()
        if project_status:
            lines.append(f"Project status: {project_status}")
        for item in (payload.get("repair_tasks") or [])[:4]:
            lines.append(f"- Repair: {item}")
        return "\n".join(lines) if lines else "[stored implementation review is empty]"

    def _error_result(self, thread_id: str, worker_task: str, user_language: str, exc: Exception) -> dict:
        code = getattr(exc, "code", "research_llm_error")
        status_code = getattr(exc, "status_code", None)
        details = str(exc).strip() or code
        if user_language == "de":
            reply = (
                "Research-Agent Fehler:\n"
                f"- Modell: `{self.model}`\n"
                f"- Fehlercode: `{code}`\n"
                f"- Details: {details}"
            )
            if status_code is not None:
                reply += f"\n- HTTP-Status: `{status_code}`"
        else:
            reply = (
                "Research agent error:\n"
                f"- Model: `{self.model}`\n"
                f"- Error code: `{code}`\n"
                f"- Details: {details}"
            )
            if status_code is not None:
                reply += f"\n- HTTP status: `{status_code}`"
        return {
            "status": "error",
            "reply": reply,
            "user_reply": reply,
            "tool_results": [],
            "internal_summary": f"Research agent failed with {code}.",
            "internal_payload": {
                "language": "en",
                "thread_id": thread_id,
                "task": f"research: {worker_task}",
                "policy": "Research is read-only and analytical. Use English for internal coordination.",
                "source": "error",
                "model": self.model,
                "error": {
                    "code": code,
                    "message": details,
                    "status_code": status_code,
                    "exception_type": type(exc).__name__,
                    "is_llm_error": isinstance(exc, LLMServiceError),
                },
            },
        }

    def _render_reply(self, payload: dict, user_language: str) -> str:
        findings = payload.get("findings") or []
        assumptions = payload.get("assumptions") or []
        open_questions = payload.get("open_questions") or []
        recommendation = payload.get("recommendation", "").strip()
        user_reply = payload["user_reply"].strip()

        sections = [user_reply]
        if findings:
            label = "Kernpunkte" if user_language == "de" else "Key findings"
            sections.append(self._format_bullets(label, findings))
        if assumptions:
            label = "Annahmen" if user_language == "de" else "Assumptions"
            sections.append(self._format_bullets(label, assumptions))
        if open_questions:
            label = "Offene Fragen" if user_language == "de" else "Open questions"
            sections.append(self._format_bullets(label, open_questions))
        if recommendation:
            prefix = "Empfehlung" if user_language == "de" else "Recommendation"
            sections.append(f"{prefix}: {recommendation}")
        return "\n\n".join(section for section in sections if section.strip())

    def _format_bullets(self, title: str, items: list[str]) -> str:
        lines = [f"{title}:"]
        lines.extend(f"- {item}" for item in items[:4])
        return "\n".join(lines)
