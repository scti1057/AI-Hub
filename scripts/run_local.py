from ai_hub.agents.manager import ManagerAgent


def main():
    agent = ManagerAgent()

    user_task = """
Build a first roadmap for a local multi-agent system with a manager, coding agent,
and research agent. The system should later connect to Discord and use local models via Ollama.
""".strip()

    result = agent.run(user_task)

    print("\n=== MANAGER RESPONSE ===\n")
    print(result)


if __name__ == "__main__":
    main()
