from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field

class MemoryItem(BaseModel):
    """Memory item stored in FAISS index and memory.json.
    
    Represents a fact, preference, tool outcome, or temporary note.
    May be embedded (for vector search) or not (scratchpad).
    """
    id: str
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str
    value: dict
    artifact_id: Optional[str] = None
    embedding: list[float] | None = None
    source: str 
    run_id: Optional[str] = None
    goal_id: Optional[str] = None
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=datetime.now)

class Artifact(BaseModel):
    """Metadata for a stored artifact blob.
    
    Immutable metadata paired with binary content in artifact store.
    """
    id: str
    content_type: str
    size_bytes: int
    source: str
    descriptor: str

class Goal(BaseModel):
    """A single bounded goal within a decomposed query.
    
    Updated by perception layer across iterations to track satisfaction.
    """
    id: str
    text: str
    done: bool = False
    attach_artifact_id: Optional[str] = None

class Observation(BaseModel):
    """Output from perception layer.
    
    Contains goal list (updated each iteration) and helper properties
    for tracking progress.
    """
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
    """Tool invocation: name and arguments.
    
    Represents a single tool to be called via MCP server.
    """
    name: str
    arguments: dict = Field(default_factory=dict)

class DecisionOutput(BaseModel):
    """Output from decision layer.
    
    Either an answer string OR a tool call, never both.
    """
    answer: Optional[str] = None
    tool_call: Optional[ToolCall] = None

    @property
    def is_answer(self) -> bool:
        return self.answer is not None