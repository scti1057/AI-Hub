import re

from ai_hub.state import ManagerDecision, RouteDecision


CODING_NOUNS = {
    "code",
    "python",
    "pytest",
    "tests",
    "debug",
    "bug",
    "fix",
    "datei",
    "file",
    "read",
    "lies",
    "workspace",
    "skript",
    "ordner",
    "unterordner",
    "verzeichnis",
    "pfad",
}

CODING_VERBS = {
    "erstelle",
    "schreibe",
    "lege",
    "lies",
    "öffne",
    "zeige",
    "starte",
    "ausführen",
    "beantrage",
    "ändere",
    "bearbeite",
    "lösche",
    "loesche",
    "delete",
    "entferne",
}

RESEARCH_KEYWORDS = {
    "research",
    "recherche",
    "analyse",
    "analyze",
    "compare",
    "vergleich",
    "dokumentation",
    "zusammenfassung",
    "quelle",
}

REVIEW_KEYWORDS = {
    "review",
    "reviewer",
    "kritik",
    "kritiker",
    "prüfe",
    "pruefe",
    "bewerte",
    "critique",
    "critic",
    "kritisch",
}

STRATEGIC_PHRASES = (
    "wie würdest du",
    "how would you",
    "strukturieren",
    "structure",
    "architektur",
    "architecture",
    "planen",
    "konzept",
    "approach",
)


class ManagerRouter:
    def decide(self, user_message: str) -> RouteDecision:
        lowered = user_message.lower()
        tokens = set(re.findall(r"\w+", lowered))

        has_coding_noun = any(keyword in tokens for keyword in CODING_NOUNS)
        has_coding_verb = any(keyword in tokens for keyword in CODING_VERBS)
        has_research_keyword = any(keyword in tokens for keyword in RESEARCH_KEYWORDS)
        has_review_keyword = any(keyword in tokens for keyword in REVIEW_KEYWORDS)
        has_file_extension = bool(re.search(r"\b[\w./-]+\.(py|txt|md|json|yaml|yml|csv)\b", lowered))
        has_workspace_phrase = "workspace" in tokens or "sandbox" in tokens
        has_execution_phrase = any(
            re.search(pattern, lowered)
            for pattern in (
                r"\bpytest\b",
                r"\bpython-datei\b",
                r"\bpython datei\b",
                r"\bausführen\b",
                r"\bstarte\b",
                r"\bbeantrage die ausführung\b",
            )
        )
        has_delete_phrase = any(
            phrase in lowered for phrase in ("lösche", "loesche", "delete", "entferne")
        )
        has_strategic_phrase = any(phrase in lowered for phrase in STRATEGIC_PHRASES)

        if has_review_keyword:
            return RouteDecision(
                decision=ManagerDecision.REVIEW,
                reason="Die Anfrage wirkt wie eine Prüfung, Kritik oder Zweitmeinung.",
            )

        if has_research_keyword:
            return RouteDecision(
                decision=ManagerDecision.RESEARCH,
                reason="Die Anfrage wirkt analysierend oder recherchelastig.",
            )

        if has_strategic_phrase and not has_execution_phrase and not has_delete_phrase:
            return RouteDecision(
                decision=ManagerDecision.DIRECT,
                reason="Die Anfrage ist strategisch oder konzeptionell und braucht keine direkte Workspace-Aktion.",
            )

        if (
            has_execution_phrase
            or has_file_extension
            or has_delete_phrase
            or (has_coding_noun and has_coding_verb)
            or (has_workspace_phrase and has_coding_verb)
        ):
            return RouteDecision(
                decision=ManagerDecision.CODING,
                reason="Die Anfrage wirkt code- oder workspace-bezogen.",
            )

        return RouteDecision(
            decision=ManagerDecision.DIRECT,
            reason="Die Anfrage kann der Manager aktuell direkt und konservativ beantworten.",
        )
