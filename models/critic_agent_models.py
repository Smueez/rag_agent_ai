
from pydantic import BaseModel, Field
from typing import List

from models.planner_agent_response_model import ChunkModel


class RankedChunk(BaseModel):
    """A chunk with its re-ranking score and reasoning."""
    query: str = Field(..., description="The query used for re-ranking")
    chunk: ChunkModel = Field(..., description="The original chunk with metadata")
    relevance_score: float = Field(..., description="Re-ranking relevance score 0 to 1")
    relevance_reason: str = Field(..., description="Brief reason why this chunk is relevant")


class ReRankingResponseModel(BaseModel):
    """Re-Ranking Agent output."""

    rewritten_query: str = Field(..., description="The query used for re-ranking")
    ranked_chunks: List[RankedChunk] = Field(..., description="Re-ranked and filtered chunks, best first")

class ReRankingInputModel(BaseModel):
    """Re-Ranking Agent input."""

    query: str = Field(..., description="The query used for re-ranking")
    chunks: List[ChunkModel] = Field(..., description="The original chunks with metadata")

class ResponseGeneratorModel(BaseModel):
    """Response Generator Agent output."""

    query: str = Field(..., description="The query that was answered")
    response: str = Field(..., description="The generated answer based strictly on context")

class ReturnedGeneratorModel(BaseModel):

    query: str = Field(..., description="The query that was answered")
    response: str = Field(..., description="The generated answer based strictly on context")
    chunks: List[RankedChunk] = Field(..., description="The original chunks with metadata")