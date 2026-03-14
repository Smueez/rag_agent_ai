from typing import List, Optional, Any

from pydantic import Field, BaseModel

from constants.enums import PlannerAgentActions

class ChunkModel(BaseModel):
    """Represents a single retrieved context chunk with metadata."""

    text: str = Field(..., description="The actual chunk content")
    score: float = Field(..., description="Relevance score from vector search, 0 to 1")
    chunk_index: str = Field(..., description="Position of this chunk in the document")
    page: Optional[int] = Field(None, description="Page number in the source document")
    source: Optional[str] = Field(None, description="Source document or book name or path")


class PlannerAgentResponseModel(BaseModel):
    """ Planner Agent Response Model """

    action: PlannerAgentActions = Field(..., description="Action type of the agent")
    rewritten_query: Optional[str] | None = Field(..., description="Rewritten query of the agent")
    response: Optional[str] | None = Field(..., description="Response of the agent")
    top_k_chunks: List[ChunkModel] = Field(default=[], description="Top K chunks returned by the agent")
    # reason: str = Field(..., description="Reason of the agent")