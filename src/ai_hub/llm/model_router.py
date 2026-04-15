from ai_hub.config import (
    MANAGER_MODEL,
    CODING_MODEL,
    RESEARCH_MODEL,
    REVIEWER_MODEL,
)


class ModelRouter:
    @staticmethod
    def get_model_for_role(role: str) -> str:
        role = role.lower().strip()

        if role == "manager":
            return MANAGER_MODEL
        if role == "coding":
            return CODING_MODEL
        if role == "research":
            return RESEARCH_MODEL
        if role == "reviewer":
            return REVIEWER_MODEL

        raise ValueError(f"Unknown role: {role}")