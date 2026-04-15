import hashlib
import logging
import threading
from time import monotonic
import requests

from ai_hub.config import OLLAMA_BASE_URL
from ai_hub.logging_config import log_event, setup_logging


logger = logging.getLogger(__name__)
setup_logging()


class LLMServiceError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        model: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.model = model
        self.status_code = status_code


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def generate(self, model: str, prompt: str, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/api/generate"
        started = monotonic()
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        completion_event = threading.Event()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        log_event(
            logger,
            "ollama_generate",
            model=model,
            base_url=self.base_url,
            prompt_chars=len(prompt),
            prompt_digest=prompt_digest,
            temperature=temperature,
            timeout_seconds=300,
        )

        def emit_waiting_logs() -> None:
            for wait_seconds in (30, 60, 120, 240):
                if completion_event.wait(wait_seconds):
                    return
                log_event(
                    logger,
                    "ollama_generate_still_waiting",
                    model=model,
                    prompt_digest=prompt_digest,
                    elapsed_ms=int((monotonic() - started) * 1000),
                )

        watchdog = threading.Thread(target=emit_waiting_logs, name="ollama-generate-watchdog", daemon=True)
        watchdog.start()
        try:
            response = requests.post(url, json=payload, timeout=300)
        except requests.Timeout as exc:
            completion_event.set()
            log_event(
                logger,
                "ollama_generate_timeout",
                model=model,
                prompt_digest=prompt_digest,
                error=str(exc),
                elapsed_ms=int((monotonic() - started) * 1000),
            )
            raise LLMServiceError(
                f"Ollama request timed out: {exc}",
                code="ollama_timeout",
                model=model,
            ) from exc
        except requests.RequestException as exc:
            completion_event.set()
            log_event(
                logger,
                "ollama_generate_unavailable",
                model=model,
                prompt_digest=prompt_digest,
                error=str(exc),
                elapsed_ms=int((monotonic() - started) * 1000),
            )
            raise LLMServiceError(
                f"Ollama request failed: {exc}",
                code="ollama_unavailable",
                model=model,
            ) from exc

        if not response.ok:
            completion_event.set()
            log_event(
                logger,
                "ollama_generate_failed",
                model=model,
                prompt_digest=prompt_digest,
                status_code=response.status_code,
                elapsed_ms=int((monotonic() - started) * 1000),
            )
            raise LLMServiceError(
                f"Ollama request failed: {response.status_code}\n{response.text}",
                code="ollama_http_error",
                model=model,
                status_code=response.status_code,
            )

        data = response.json()
        completion_event.set()
        response_text = data.get("response", "").strip()
        if not response_text:
            raise LLMServiceError(
                "Ollama returned an empty response.",
                code="ollama_empty_response",
                model=model,
                status_code=response.status_code,
            )
        log_event(
            logger,
            "ollama_generate_completed",
            model=model,
            prompt_digest=prompt_digest,
            status_code=response.status_code,
            elapsed_ms=int((monotonic() - started) * 1000),
            response_chars=len(response_text),
        )
        return response_text

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            log_event(logger, "ollama_availability", base_url=self.base_url, ok=response.ok)
            return response.ok
        except requests.RequestException:
            log_event(logger, "ollama_availability_failed", base_url=self.base_url)
            return False
