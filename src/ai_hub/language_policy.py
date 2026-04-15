import re
from dataclasses import dataclass


@dataclass(slots=True)
class LanguageContext:
    user_language: str
    user_message: str
    internal_message: str


class LanguagePolicy:
    SUPPORTED_USER_LANGUAGES = {"de", "en"}

    def build_context(self, user_message: str) -> LanguageContext:
        user_language = self.detect_user_language(user_message)
        internal_message = self.normalize_for_internal_agents(user_message, user_language)
        return LanguageContext(
            user_language=user_language,
            user_message=user_message,
            internal_message=internal_message,
        )

    def detect_user_language(self, user_message: str) -> str:
        lowered = user_message.lower()
        german_markers = {
            "bitte",
            "und",
            "oder",
            "was",
            "kannst",
            "aktuell",
            "ausführung",
            "freigabe",
            "datei",
            "inhalt",
            "lies",
            "erstelle",
        }
        if any(marker in lowered for marker in german_markers):
            return "de"
        return "en"

    def normalize_for_internal_agents(self, user_message: str, user_language: str) -> str:
        if user_language == "en":
            return user_message.strip()

        normalized = user_message.strip()
        replacements = [
            (r"\bBitte\b", "Please"),
            (r"\bbitte\b", "please"),
            (r"\berstelle\b", "create"),
            (r"\bErstelle\b", "Create"),
            (r"\blege an\b", "create"),
            (r"\bschreibe\b", "write"),
            (r"\bDatei\b", "file"),
            (r"\bdatei\b", "file"),
            (r"\bOrdner\b", "directory"),
            (r"\bordner\b", "directory"),
            (r"\bVerzeichnis\b", "directory"),
            (r"\bworkspace\b", "workspace"),
            (r"\bmit dem Inhalt\b", "with content"),
            (r"\bInhalt\b", "content"),
            (r"\bausgibt\b", "that prints"),
            (r"\bbeantrage anschließend die Ausführung\b", "then request execution"),
            (r"\bbeantrage die Ausführung\b", "request execution"),
            (r"\bAusführung\b", "execution"),
            (r"\bausführen\b", "run"),
            (r"\bstarte\b", "run"),
            (r"\bLies\b", "Read"),
            (r"\blies\b", "read"),
            (r"\bzeige\b", "show"),
            (r"\bkleine\b", "small"),
        ]
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)
        return normalized

    def user_text(self, language: str, german_text: str, english_text: str) -> str:
        if language == "de":
            return german_text
        return english_text
