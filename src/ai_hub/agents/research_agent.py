from ai_hub.memory.history import format_thread_history


class ResearchAgent:
    role = "research"

    def handle_task(
        self,
        thread_id: str,
        user_task: str,
        history: list[dict],
        internal_task: str | None = None,
    ) -> dict:
        summary = format_thread_history(history, limit=4)
        reply = (
            "Research-Agent im Analysemodus:\n"
            "- Ich arbeite aktuell bewusst lesend und vorbereitend.\n"
            "- Ich würde als Nächstes Fragen, Quellenbedarf und offene Annahmen strukturieren.\n"
            f"- Relevanter Thread-Kontext:\n{summary}"
        )
        return {
            "reply": reply,
            "internal_payload": {
                "language": "en",
                "task": internal_task or user_task,
                "policy": "Research remains analytical and read-oriented.",
            },
        }
