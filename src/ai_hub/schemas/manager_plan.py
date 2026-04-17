from typing import Literal

from pydantic import BaseModel, Field

from ai_hub.schemas.coding_delegation import CodingDelegationPlan


class ProjectOutline(BaseModel):
    summary: str = Field(min_length=1)
    repo_structure: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    validation_steps: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    autonomous_execution: bool = False


class ManagerPlan(BaseModel):
    summary: str = Field(min_length=1)
    decision: Literal["direct", "plan", "coding", "research", "review"]
    reason: str = Field(min_length=1)
    user_reply: str = Field(min_length=1)
    internal_task_for_worker: str = Field(default="")
    approval_needed: bool = False
    coding_plan: CodingDelegationPlan | None = None
    project_outline: ProjectOutline | None = None
