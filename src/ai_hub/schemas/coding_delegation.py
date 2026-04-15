from pydantic import BaseModel, Field

from ai_hub.schemas.coding_actions import CodingActionBatch


class CodingDelegationPlan(BaseModel):
    summary: str = Field(default="")
    rationale: str = Field(default="")
    approval_needed: bool = False
    actions: CodingActionBatch = Field(default_factory=CodingActionBatch)
