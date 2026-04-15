import logging
import requests

from ai_hub.config import OLLAMA_BASE_URL
from ai_hub.logging_config import log_event, setup_logging


logger = logging.getLogger(__name__)
setup_logging()


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def generate(self, model: str, prompt: str, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        log_event(logger, "ollama_generate", model=model, base_url=self.base_url)
        response = requests.post(url, json=payload, timeout=300)

        if not response.ok:
            log_event(logger, "ollama_generate_failed", model=model, status_code=response.status_code)
            raise RuntimeError(
                f"Ollama request failed: {response.status_code}\n{response.text}"
            )

        data = response.json()
        return data.get("response", "").strip()

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            log_event(logger, "ollama_availability", base_url=self.base_url, ok=response.ok)
            return response.ok
        except requests.RequestException:
            log_event(logger, "ollama_availability_failed", base_url=self.base_url)
            return False
