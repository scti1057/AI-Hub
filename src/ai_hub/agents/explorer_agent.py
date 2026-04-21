import hashlib
import json
import logging
import re
from pathlib import Path

from ai_hub.config import EXPLORER_DEBUG_LOG_MAX_CHARS, EXPLORER_DEBUG_LOG_PROMPTS
from ai_hub.language_policy import LanguagePolicy
from ai_hub.logging_config import log_event, log_text_block, setup_logging
from ai_hub.llm.model_router import ModelRouter
from ai_hub.llm.ollama_client import LLMServiceError, OllamaClient
from ai_hub.memory.history import format_thread_history
from ai_hub.memory.store import HubStore
from ai_hub.tools import file_tools


logger = logging.getLogger(__name__)
setup_logging()

IGNORED_DIR_NAMES = {
    ".git",
    ".next",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
}
TEXT_SUFFIXES = {
    ".py",
    ".tsx",
    ".ts",
    ".jsx",
    ".js",
    ".json",
    ".md",
    ".mdx",
    ".css",
    ".html",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".txt",
}


class ExplorerAgent:
    role = "explorer"

    def __init__(
        self,
        language_policy: LanguagePolicy | None = None,
        client: OllamaClient | None = None,
        store: HubStore | None = None,
    ) -> None:
        self.language_policy = language_policy or LanguagePolicy()
        self.client = client or OllamaClient()
        self.store = store
        self.model = ModelRouter.get_model_for_role("explorer")
        self.system_prompt = self._load_system_prompt()

    def handle_task(
        self,
        thread_id: str,
        user_task: str,
        history: list[dict],
        internal_task: str | None = None,
    ) -> dict:
        language_context = self.language_policy.build_context(user_task)
        history_summary = format_thread_history(history, limit=6)
        worker_task = internal_task or language_context.internal_message
        workspace = file_tools.ensure_thread_workspace(thread_id)
        workspace_index = self._workspace_index(workspace)
        workspace_tree = self._workspace_tree(workspace_index)
        candidate_paths = self._derive_candidate_paths(workspace_index, user_task, worker_task)
        candidate_snapshots = self._capture_snapshots(thread_id, candidate_paths, limit=8)
        structure_signals = self._structure_signals(workspace_index)
        artifact_context = self._load_artifact_context(thread_id)
        validation_candidate_paths = self._derive_validation_candidate_paths(
            workspace_index=workspace_index,
            candidate_paths=candidate_paths,
            artifact_context=artifact_context,
            user_task=user_task,
            worker_task=worker_task,
        )

        try:
            prompt = self._build_prompt(
                history_summary=history_summary,
                user_task=user_task,
                internal_task=worker_task,
                user_language=language_context.user_language,
                workspace_tree=workspace_tree,
                candidate_context=self._candidate_context_block(candidate_snapshots),
                structure_signals=structure_signals,
                artifact_memory=self._artifact_memory_block(artifact_context),
                latest_constraints_memory=self._latest_constraints_block(artifact_context),
                latest_change_memory=self._latest_change_block(artifact_context),
                latest_review_memory=self._latest_review_block(artifact_context),
            )
            prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            log_event(
                logger,
                "explorer_ollama_request",
                thread_id=thread_id,
                model=self.model,
                history_chars=len(history_summary),
                worker_task_chars=len(worker_task),
                prompt_chars=len(prompt),
                prompt_digest=prompt_digest,
                worker_task_preview=worker_task[:160],
            )
            if EXPLORER_DEBUG_LOG_PROMPTS:
                log_text_block(
                    logger,
                    "explorer_prompt_body",
                    prompt,
                    max_chars=EXPLORER_DEBUG_LOG_MAX_CHARS,
                    thread_id=thread_id,
                    model=self.model,
                    prompt_digest=prompt_digest,
                )
            raw_response = self.client.generate(model=self.model, prompt=prompt, temperature=0.1)
            log_event(
                logger,
                "explorer_generate_completed",
                thread_id=thread_id,
                model=self.model,
                prompt_digest=prompt_digest,
                raw_response_chars=len(raw_response),
            )
            if EXPLORER_DEBUG_LOG_PROMPTS:
                log_text_block(
                    logger,
                    "explorer_raw_response",
                    raw_response,
                    max_chars=EXPLORER_DEBUG_LOG_MAX_CHARS,
                    thread_id=thread_id,
                    model=self.model,
                    prompt_digest=prompt_digest,
                )
            parsed = self._parse_response(raw_response)
            relevant_paths = self._normalize_relevant_paths(
                thread_id=thread_id,
                requested_paths=parsed.get("relevant_paths") or [],
                workspace_index=workspace_index,
                user_task=user_task,
                worker_task=worker_task,
                candidate_paths=candidate_paths,
            )
            validation_relevant_paths = self._normalize_validation_paths(
                requested_paths=parsed.get("validation_relevant_paths") or [],
                workspace_index=workspace_index,
                fallback_paths=validation_candidate_paths,
            )
            snapshots = self._capture_snapshots(thread_id, relevant_paths)
            reply = self._render_reply(parsed, language_context.user_language)
            log_event(
                logger,
                "explorer_ollama_response",
                thread_id=thread_id,
                model=self.model,
                prompt_digest=prompt_digest,
                summary_chars=len(parsed["summary"]),
                repo_overview_chars=len(parsed["repo_overview"]),
                relevant_paths_count=len(relevant_paths),
                validation_paths_count=len(validation_relevant_paths),
                reply_chars=len(reply),
            )
        except Exception as exc:
            log_event(
                logger,
                "explorer_error",
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
                "task": f"explore: {worker_task}",
                "policy": "Explorer is read-only. Build a compact repo context for one coding step.",
                "source": "ollama",
                "model": self.model,
                "summary": parsed["summary"],
                "repo_overview": parsed["repo_overview"],
                "relevant_paths": relevant_paths,
                "validation_relevant_paths": validation_relevant_paths,
                "suggested_definition_of_done": parsed["suggested_definition_of_done"],
                "suggested_checks": parsed["suggested_checks"],
                "risks": parsed["risks"],
                "snapshots": snapshots,
                "candidate_paths": candidate_paths,
                "validation_candidate_paths": validation_candidate_paths,
                "candidate_snapshots": candidate_snapshots,
                "structure_signals": structure_signals,
                "artifact_context": artifact_context,
                "workspace_tree": workspace_tree,
                "workspace_file_count": len(workspace_index),
                "raw_response": raw_response,
            },
        }

    def _load_system_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "explorer_agent.txt"
        return prompt_path.read_text(encoding="utf-8").strip()

    def _build_prompt(
        self,
        *,
        history_summary: str,
        user_task: str,
        internal_task: str,
        user_language: str,
        workspace_tree: str,
        candidate_context: str,
        structure_signals: list[str],
        artifact_memory: str,
        latest_constraints_memory: str,
        latest_change_memory: str,
        latest_review_memory: str,
    ) -> str:
        structure_signal_block = "\n".join(f"- {item}" for item in structure_signals[:8]) if structure_signals else "- No strong structure signals detected."
        return f"""
{self.system_prompt}

You are working inside a local multi-agent backend.
The user-facing text must be written in this language code: {user_language}.
Internal coordination stays in English.

Recent thread context:
{history_summary}

Original user task:
{user_task}

Internal worker task:
{internal_task}

Workspace tree:
{workspace_tree}

Stored project/artifact memory:
{artifact_memory}

Confirmed user constraints and decisions:
{latest_constraints_memory}

Latest stored change snapshot:
{latest_change_memory}

Latest stored implementation review:
{latest_review_memory}

Observed structure signals:
{structure_signal_block}

Candidate file previews:
{candidate_context}

Return JSON only with this shape:
{{
  "summary": "short repo-context summary",
  "repo_overview": "compact structure summary",
  "relevant_paths": ["existing/path.ext"],
  "validation_relevant_paths": ["existing/test/or/entry/path.ext"],
  "suggested_definition_of_done": ["step-local acceptance check"],
  "suggested_checks": ["specific validation or integration check"],
  "risks": ["repo-specific risk or inconsistency"],
  "user_reply": "compact user-facing explorer answer in the user's language"
}}
""".strip()

    def _parse_response(self, response: str) -> dict:
        json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not json_match:
            raise ValueError("Explorer agent did not return JSON.")
        payload = json.loads(json_match.group(0))
        summary = str(payload.get("summary", "")).strip()
        repo_overview = str(payload.get("repo_overview", "")).strip()
        user_reply = str(payload.get("user_reply", "")).strip()
        if not summary or not repo_overview or not user_reply:
            raise ValueError("Explorer agent response is missing required fields.")
        return {
            "summary": summary,
            "repo_overview": repo_overview,
            "relevant_paths": self._normalize_list(payload.get("relevant_paths")),
            "validation_relevant_paths": self._normalize_list(payload.get("validation_relevant_paths")),
            "suggested_definition_of_done": self._normalize_list(payload.get("suggested_definition_of_done")),
            "suggested_checks": self._normalize_list(payload.get("suggested_checks")),
            "risks": self._normalize_list(payload.get("risks")),
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
        return normalized[:8]

    def _workspace_index(self, workspace: Path) -> list[str]:
        files: list[str] = []
        for path in sorted(workspace.rglob("*")):
            if not path.is_file():
                continue
            relative_parts = path.relative_to(workspace).parts
            if any(part in IGNORED_DIR_NAMES for part in relative_parts):
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"README", "Dockerfile"}:
                continue
            files.append(str(path.relative_to(workspace)).replace("\\", "/"))
        return files[:240]

    def _workspace_tree(self, workspace_index: list[str]) -> str:
        if not workspace_index:
            return "[workspace is empty]"
        return "\n".join(f"- {path}" for path in workspace_index[:180])

    def _normalize_relevant_paths(
        self,
        *,
        thread_id: str,
        requested_paths: list[str],
        workspace_index: list[str],
        user_task: str,
        worker_task: str,
        candidate_paths: list[str],
    ) -> list[str]:
        available = set(workspace_index)
        normalized: list[str] = []
        for raw_path in requested_paths:
            candidate = str(raw_path).strip().replace("\\", "/").lstrip("./")
            if candidate in available and candidate not in normalized:
                normalized.append(candidate)
        if normalized:
            return normalized[:8]

        for candidate in candidate_paths:
            if candidate in available and candidate not in normalized:
                normalized.append(candidate)
        if normalized:
            return normalized[:8]

        hints = self._path_hints_from_text(f"{user_task}\n{worker_task}")
        for candidate in workspace_index:
            if any(hint in candidate.lower() for hint in hints) and candidate not in normalized:
                normalized.append(candidate)
        if normalized:
            return normalized[:8]
        return workspace_index[:6]

    def _path_hints_from_text(self, text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9_./\\-]+", text.lower())
        hints = []
        for token in tokens:
            if "/" in token or "." in token:
                hints.append(token.strip("./"))
        return hints[:12]

    def _derive_candidate_paths(self, workspace_index: list[str], user_task: str, worker_task: str) -> list[str]:
        if not workspace_index:
            return []

        hints = self._task_keywords(f"{user_task}\n{worker_task}")
        scored: list[tuple[int, str]] = []
        for path in workspace_index:
            lowered = path.lower()
            score = 0
            for hint in hints:
                if hint in lowered:
                    score += 3
                for part in lowered.replace(".", "/").split("/"):
                    if hint == part:
                        score += 2
            if lowered.startswith("src/app/"):
                score += 1
            if lowered.endswith(("layout.tsx", "layout.js", "page.tsx", "page.js")):
                score += 1
            if score > 0:
                scored.append((score, path))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [path for _, path in scored[:10]]
        if selected:
            return selected
        return workspace_index[:8]

    def _derive_validation_candidate_paths(
        self,
        *,
        workspace_index: list[str],
        candidate_paths: list[str],
        artifact_context: dict,
        user_task: str,
        worker_task: str,
    ) -> list[str]:
        if not workspace_index:
            return []

        validation_steps = (artifact_context.get("project_plan") or {}).get("validation_steps") or []
        combined_text = "\n".join([user_task, worker_task] + [str(item) for item in validation_steps])
        hints = self._task_keywords(combined_text) + self._path_hints_from_text(combined_text)
        selected: list[str] = []
        scored: list[tuple[int, str]] = []
        for path in workspace_index:
            lowered = path.lower()
            score = 0
            if lowered.startswith("tests/") or "/tests/" in lowered:
                score += 5
            if re.search(r"(^|/)(test_[^/]+|[^/]+_test)\.py$", lowered):
                score += 4
            if any(lowered.endswith(suffix) for suffix in (".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx", ".test.ts", ".test.tsx", ".test.js", ".test.jsx")):
                score += 4
            if any(lowered.endswith(name) for name in ("src/main.py", "main.py", "app.py", "manage.py", "src/app/page.tsx", "src/app/page.js", "src/index.tsx", "src/index.jsx", "src/index.ts", "src/index.js")):
                score += 4
            if path in candidate_paths:
                score += 2
            for hint in hints[:20]:
                hint_text = str(hint).strip().lower()
                if not hint_text:
                    continue
                if hint_text in lowered:
                    score += 2
            if score > 0:
                scored.append((score, path))

        scored.sort(key=lambda item: (-item[0], item[1]))
        for _, path in scored:
            if path not in selected:
                selected.append(path)
        return selected[:8]

    def _task_keywords(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9_-]+", text.lower())
        ignored = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "step",
            "file",
            "files",
            "repo",
            "project",
            "coding",
            "implement",
            "implementation",
            "manager",
            "reviewer",
            "explorer",
            "agent",
            "task",
            "latest",
            "next",
        }
        keywords: list[str] = []
        for token in tokens:
            if len(token) < 3 or token in ignored:
                continue
            if token not in keywords:
                keywords.append(token)
        return keywords[:16]

    def _capture_snapshots(self, thread_id: str, relevant_paths: list[str], limit: int = 8) -> list[dict]:
        snapshots: list[dict] = []
        for path in relevant_paths[:limit]:
            try:
                content = file_tools.read_file(thread_id, path)
            except Exception:
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            snapshots.append(
                {
                    "path": path,
                    "preview": content[:400],
                    "bytes": len(content.encode("utf-8")),
                    "sha256": digest,
                    "exists": True,
                }
            )
        return snapshots

    def _normalize_validation_paths(
        self,
        *,
        requested_paths: list[str],
        workspace_index: list[str],
        fallback_paths: list[str],
    ) -> list[str]:
        available = set(workspace_index)
        normalized: list[str] = []
        for raw_path in requested_paths:
            candidate = str(raw_path).strip().replace("\\", "/").lstrip("./")
            if candidate in available and candidate not in normalized:
                normalized.append(candidate)
        if normalized:
            return normalized[:8]
        return [path for path in fallback_paths if path in available][:8]

    def _candidate_context_block(self, candidate_snapshots: list[dict]) -> str:
        if not candidate_snapshots:
            return "[no candidate file previews captured]"
        blocks = []
        for snapshot in candidate_snapshots[:8]:
            blocks.append(
                f"File: {snapshot['path']}\n"
                f"Preview: {snapshot['preview'][:320] or '[empty]'}"
            )
        return "\n\n".join(blocks)

    def _structure_signals(self, workspace_index: list[str]) -> list[str]:
        signals: list[str] = []
        route_variants: dict[str, set[str]] = {}
        for path in workspace_index:
            normalized = re.sub(r"\.(tsx|ts|jsx|js)$", "", path)
            route_variants.setdefault(normalized, set()).add(path.rsplit(".", 1)[-1].lower())

        duplicated_routes = [
            normalized
            for normalized, variants in route_variants.items()
            if len(variants) > 1 and any(normalized.endswith(suffix) for suffix in ("/page", "/layout"))
        ]
        for item in duplicated_routes[:4]:
            signals.append(f"Multiple route file variants exist for `{item}`.")

        if any(path.startswith("src/app/") for path in workspace_index):
            signals.append("App Router structure is present under `src/app/`.")
        if any(path.startswith("src/components/") for path in workspace_index):
            signals.append("Reusable component structure exists under `src/components/`.")
        if any(path.startswith("src/content/") for path in workspace_index):
            signals.append("Content files exist under `src/content/`.")
        return signals[:8]

    def _load_artifact_context(self, thread_id: str) -> dict:
        if self.store is None:
            return {
                "recent_artifacts": [],
                "latest_change_snapshot": {},
                "latest_review": {},
            }

        try:
            artifacts = self.store.list_artifacts(thread_id)
            latest_change = self.store.get_artifact(thread_id, "coding_change_snapshot")
            latest_review = self.store.get_artifact(thread_id, "implementation_review")
            project_state = self.store.get_artifact(thread_id, "project_state")
            project_plan = self.store.get_artifact(thread_id, "project_plan")
        except Exception:
            return {
                "recent_artifacts": [],
                "latest_change_snapshot": {},
                "latest_review": {},
                "manager_constraints": {},
                "project_state": {},
                "project_plan": {},
            }

        preferred = (
            "project_state",
            "project_plan",
            "manager_constraints",
            "change_request_brief",
            "project_brief",
            "coding_step_contract",
            "coding_change_snapshot",
            "coding_context_snapshot",
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
            "recent_artifacts": recent_artifacts[:6],
            "latest_change_snapshot": (latest_change or {}).get("content", {}),
            "latest_review": (latest_review or {}).get("content", {}),
            "manager_constraints": ((self.store.get_artifact(thread_id, "manager_constraints") or {}).get("content", {})),
            "project_state": (project_state or {}).get("content", {}),
            "project_plan": (project_plan or {}).get("content", {}),
        }

    def _artifact_memory_block(self, artifact_context: dict) -> str:
        recent_artifacts = artifact_context.get("recent_artifacts") or []
        lines = [
            f"- {item.get('kind', 'artifact')}: {item.get('summary', '') or '[no summary]'}"
            for item in recent_artifacts[:6]
        ]
        project_state = artifact_context.get("project_state") or {}
        validation_status = str(project_state.get("validation_status", "")).strip()
        validation_summary = str(project_state.get("validation_summary", "")).strip()
        if validation_status:
            lines.append(f"- project_state.validation_status: {validation_status}")
        if validation_summary:
            lines.append(f"- project_state.validation_summary: {validation_summary}")
        project_plan = artifact_context.get("project_plan") or {}
        for step in (project_plan.get("validation_steps") or [])[:4]:
            lines.append(f"- project_plan.validation_step: {step}")
        if not lines:
            return "[no stored artifact memory available]"
        return "\n".join(lines)

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
        code = getattr(exc, "code", "explorer_llm_error")
        status_code = getattr(exc, "status_code", None)
        details = str(exc).strip() or code
        if user_language == "de":
            reply = (
                "Explorer-Agent Fehler:\n"
                f"- Modell: `{self.model}`\n"
                f"- Fehlercode: `{code}`\n"
                f"- Details: {details}"
            )
            if status_code is not None:
                reply += f"\n- HTTP-Status: `{status_code}`"
        else:
            reply = (
                "Explorer agent error:\n"
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
            "internal_summary": f"Explorer agent failed with {code}.",
            "internal_payload": {
                "language": "en",
                "thread_id": thread_id,
                "task": f"explore: {worker_task}",
                "policy": "Explorer is read-only. Build a compact repo context for one coding step.",
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
        if payload["relevant_paths"]:
            title = "Relevante Dateien" if user_language == "de" else "Relevant files"
            sections.append(self._format_bullets(title, payload["relevant_paths"]))
        if payload.get("validation_relevant_paths"):
            title = "Validierungsdateien" if user_language == "de" else "Validation files"
            sections.append(self._format_bullets(title, payload["validation_relevant_paths"]))
        if payload["suggested_definition_of_done"]:
            title = "Empfohlene Done-Kriterien" if user_language == "de" else "Suggested done criteria"
            sections.append(self._format_bullets(title, payload["suggested_definition_of_done"]))
        return "\n\n".join(section for section in sections if section.strip())

    def _format_bullets(self, title: str, items: list[str]) -> str:
        return "\n".join([f"{title}:"] + [f"- {item}" for item in items[:4]])
