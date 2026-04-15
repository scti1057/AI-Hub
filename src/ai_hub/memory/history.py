def format_thread_history(messages: list[dict], limit: int = 8) -> str:
    trimmed = messages[-limit:]
    if not trimmed:
        return "Noch kein Verlauf."

    lines = []
    for item in trimmed:
        role = item["role"]
        speaker = item.get("agent") or role
        lines.append(f"{speaker}: {item['content']}")
    return "\n".join(lines)
