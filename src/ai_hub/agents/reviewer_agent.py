import json
import hashlib
import logging
import re
from pathlib import Path
from time import monotonic

from ai_hub.config import REVIEWER_DEBUG_LOG_MAX_CHARS, REVIEWER_DEBUG_LOG_PROMPTS
from ai_hub.language_policy import LanguagePolicy
from ai_hub.logging_config import log_event, log_text_block, setup_logging
from ai_hub.llm.model_router import ModelRouter
from ai_hub.llm.ollama_client import LLMServiceError, OllamaClient
from ai_hub.memory.history import format_thread_history


logger = logging.getLogger(__name__)
setup_logging()


class ReviewerAgent:
    role = "review"

    def __init__(
        self,
        language_policy: LanguagePolicy | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        self.language_policy = language_policy or LanguagePolicy()
        self.client = client or OllamaClient()
        self.model = ModelRouter.get_model_for_role("reviewer")
        self.system_prompt = self._load_system_prompt()

    def handle_task(
        self,
        thread_id: str,
        user_task: str,
        history: list[dict],
        internal_task: str | None = None,
    ) -> dict:
        language_context = self.language_policy.build_context(user_task)
        summary = format_thread_history(history, limit=6)
        worker_task = internal_task or language_context.internal_message
        started = monotonic()

        try:
            prompt = self._build_prompt(summary, user_task, worker_task, language_context.user_language)
            prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            log_event(
                logger,
                "reviewer_ollama_request",
                thread_id=thread_id,
                model=self.model,
                history_chars=len(summary),
                worker_task_chars=len(worker_task),
                prompt_chars=len(prompt),
                prompt_digest=prompt_digest,
                worker_task_preview=worker_task[:160],
            )
            if REVIEWER_DEBUG_LOG_PROMPTS:
                log_event(
                    logger,
                    "reviewer_prompt_sections",
                    thread_id=thread_id,
                    model=self.model,
                    prompt_digest=prompt_digest,
                    system_prompt_chars=len(self.system_prompt),
                    history_summary_chars=len(summary),
                    user_task_chars=len(user_task),
                    internal_task_chars=len(worker_task),
                    user_language=language_context.user_language,
                )
                log_text_block(
                    logger,
                    "reviewer_prompt_body",
                    prompt,
                    max_chars=REVIEWER_DEBUG_LOG_MAX_CHARS,
                    thread_id=thread_id,
                    model=self.model,
                    prompt_digest=prompt_digest,
                )
            log_event(
                logger,
                "reviewer_generate_started",
                thread_id=thread_id,
                model=self.model,
                prompt_digest=prompt_digest,
            )
            raw_response = self.client.generate(model=self.model, prompt=prompt, temperature=0.2)
            log_event(
                logger,
                "reviewer_generate_completed",
                thread_id=thread_id,
                model=self.model,
                prompt_digest=prompt_digest,
                raw_response_chars=len(raw_response),
            )
            if REVIEWER_DEBUG_LOG_PROMPTS:
                log_text_block(
                    logger,
                    "reviewer_raw_response",
                    raw_response,
                    max_chars=REVIEWER_DEBUG_LOG_MAX_CHARS,
                    thread_id=thread_id,
                    model=self.model,
                    prompt_digest=prompt_digest,
                )
            log_event(
                logger,
                "reviewer_parse_started",
                thread_id=thread_id,
                model=self.model,
                prompt_digest=prompt_digest,
            )
            parsed = self._parse_response(raw_response)
            log_event(
                logger,
                "reviewer_parse_completed",
                thread_id=thread_id,
                model=self.model,
                prompt_digest=prompt_digest,
                summary_chars=len(parsed["summary"]),
                findings_count=len(parsed["findings"]),
                open_questions_count=len(parsed["open_questions"]),
            )
        except Exception as exc:
            log_event(
                logger,
                "reviewer_error",
                thread_id=thread_id,
                model=self.model,
                error=str(exc),
                elapsed_ms=int((monotonic() - started) * 1000),
            )
            return self._error_result(thread_id, worker_task, language_context.user_language, exc)

        log_event(
            logger,
            "reviewer_ollama_response",
            thread_id=thread_id,
            model=self.model,
            elapsed_ms=int((monotonic() - started) * 1000),
            summary_chars=len(parsed["summary"]),
            findings_count=len(parsed["findings"]),
            open_questions_count=len(parsed["open_questions"]),
            recommendation_chars=len(parsed["recommendation"]),
        )

        reply = self._render_reply(parsed, language_context.user_language)
        log_event(
            logger,
            "reviewer_returning_result",
            thread_id=thread_id,
            model=self.model,
            prompt_digest=prompt_digest,
            reply_chars=len(reply),
        )
        return {
            "status": "completed",
            "reply": reply,
            "user_reply": reply,
            "tool_results": [],
            "internal_summary": parsed["summary"],
            "internal_payload": {
                "language": "en",
                "thread_id": thread_id,
                "task": f"review: {worker_task}",
                "policy": "Review is analytical and read-only. Focus on risks, gaps, and next checks.",
                "source": "ollama",
                "model": self.model,
                "summary": parsed["summary"],
                "findings": parsed["findings"],
                "assumptions": parsed["assumptions"],
                "open_questions": parsed["open_questions"],
                "recommendation": parsed["recommendation"],
                "raw_response": raw_response,
            },
        }

    def _load_system_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "reviewer_agent.txt"
        return prompt_path.read_text(encoding="utf-8").strip()

    def _build_prompt(
        self,
        history_summary: str,
        user_task: str,
        internal_task: str,
        user_language: str,
    ) -> str:
        return f"""
{self.system_prompt}

You are working inside a local multi-agent backend.
You are the critical reviewer, not the coding or manager agent.
Stay read-only and analytical.
The user-facing text must be written in this language code: {user_language}.
Internal coordination stays in English.

Recent thread context:
{history_summary}

Original user task:
{user_task}

Internal worker task:
{internal_task}

Return JSON only with this shape:
{{
  "summary": "short review summary",
  "findings": ["concrete risk, flaw, or concern"],
  "assumptions": ["assumption or context gap"],
  "open_questions": ["question to resolve next"],
  "recommendation": "best next step in the user's language",
  "user_reply": "compact user-facing review reply in the user's language"
}}
""".strip()

    def _parse_response(self, response: str) -> dict:
        json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not json_match:
            raise ValueError("Reviewer agent did not return JSON.")
        payload = json.loads(json_match.group(0))
        summary = str(payload.get("summary", "")).strip()
        user_reply = str(payload.get("user_reply", "")).strip()
        if not summary or not user_reply:
            raise ValueError("Reviewer agent response is missing required fields.")
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
        return [text for item in value if (text := str(item).strip())][:6]

    def _error_result(self, thread_id: str, worker_task: str, user_language: str, exc: Exception) -> dict:
        code = getattr(exc, "code", "reviewer_llm_error")
        status_code = getattr(exc, "status_code", None)
        details = str(exc).strip() or code
        if user_language == "de":
            reply = (
                "Kritiker-Agent Fehler:\n"
                f"- Modell: `{self.model}`\n"
                f"- Fehlercode: `{code}`\n"
                f"- Details: {details}"
            )
            if status_code is not None:
                reply += f"\n- HTTP-Status: `{status_code}`"
        else:
            reply = (
                "Reviewer agent error:\n"
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
            "internal_summary": f"Reviewer agent failed with {code}.",
            "internal_payload": {
                "language": "en",
                "thread_id": thread_id,
                "task": f"review: {worker_task}",
                "policy": "Review is analytical and read-only. Focus on risks, gaps, and next checks.",
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
        sections = [payload["user_reply"].strip()]
        if payload["findings"]:
            title = "Kritische Punkte" if user_language == "de" else "Findings"
            sections.append(self._format_bullets(title, payload["findings"]))
        if payload["assumptions"]:
            title = "Annahmen" if user_language == "de" else "Assumptions"
            sections.append(self._format_bullets(title, payload["assumptions"]))
        if payload["open_questions"]:
            title = "Offene Fragen" if user_language == "de" else "Open questions"
            sections.append(self._format_bullets(title, payload["open_questions"]))
        if payload["recommendation"]:
            prefix = "Empfehlung" if user_language == "de" else "Recommendation"
            sections.append(f"{prefix}: {payload['recommendation']}")
        return "\n\n".join(section for section in sections if section.strip())

    def _format_bullets(self, title: str, items: list[str]) -> str:
        return "\n".join([f"{title}:"] + [f"- {item}" for item in items[:4]])
