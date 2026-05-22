from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field

class MemoryItem(BaseModel):
    id: str
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str
    value: dict
    artifact_id: Optional[str] = None
    source: str 
    run_id: Optional[str] = None
    goal_id: Optional[str] = None
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=datetime.now)

class Artifact(BaseModel):
    id: str
    content_type: str
    size_bytes: int
    source: str
    descriptor: str

class Goal(BaseModel):
    id: str
    text: str
    done: bool = False
    attach_artifact_id: Optional[str] = None

class Observation(BaseModel):
    goals: list[Goal]

    @property
    def all_done(self) -> bool:
        return all(goal.done for goal in self.goals)
    
    def next_unfinished(self) -> Optional[Goal]:
        for goal in self.goals:
            if not goal.done:
                return goal
        return None
    
class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)

class DecisionOutput(BaseModel):
    answer: Optional[str] = None
    tool_call: Optional[ToolCall] = None

    @property
    def is_answer(self) -> bool:
        return self.answer is not None