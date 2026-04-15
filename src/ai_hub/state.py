from dataclasses import dataclass
from enum import StrEnum


class ThreadStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ManagerDecision(StrEnum):
    DIRECT = "direct"
    CODING = "coding"
    RESEARCH = "research"


@dataclass(slots=True)
class RouteDecision:
    decision: ManagerDecision
    reason: str
