
from pydantic import BaseModel, Field
from typing import List, Optional

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

class CriticFeedback(BaseModel):
    """Specific feedback on what needs to change."""

    unsupported_claims: List[str] = Field(..., description="Claims that had no chunk support")
    suggested_fix: str = Field(..., description="Specific instruction for the Generator on how to fix")
    missing_context: Optional[str] = Field(None, description="What context would be needed to support the claims")


class ResponseGeneratorModel(BaseModel):
    """Response Generator Agent output."""

    query: str = Field(..., description="The query that was answered")
    response: str = Field(..., description="The generated answer based strictly on context")

class ReturnedGeneratorModel(BaseModel):

    query: str = Field(..., description="The query that was answered")
    response: str = Field(..., description="The generated answer based strictly on context")
    chunks: List[RankedChunk] = Field(..., description="The original chunks with metadata")

class CitationModel(BaseModel):
    """A single claim mapped to its supporting chunks."""

    claim: str = Field(..., description="A single sentence or claim from the response")
    supporting_chunk_indices: List[int] = Field(..., description="chunk_index values that support this claim")
    confidence: float = Field(..., description="Confidence that this claim is supported, 0 to 1")


class GroundingResponseModel(BaseModel):
    """Grounding Agent output."""

    query: str = Field(..., description="The original query")
    response: str = Field(..., description="The response being grounded, unchanged")
    citations: List[CitationModel] = Field(..., description="Every claim mapped to supporting chunks")
    unsupported_claims: List[str] = Field(default=[], description="Claims with no chunk support found")
    is_fully_grounded: bool = Field(..., description="True only if unsupported_claims is empty")
    grounding_score: float = Field(..., description="Overall grounding score 0 to 1")

class CriticResponseModel(BaseModel):
    """Critic Agent output."""

    attempt: int = Field(..., description="Which attempt this is, starting from 1")
    verdict: str = Field(..., description="Either 'approved' or 'rejected'")
    grounding_score: float = Field(..., description="Grounding score from the Grounding Agent")
    feedback: Optional[CriticFeedback] = Field(None, description="Populated only on rejection")
    final_response: Optional[str] = Field(None, description="Populated only on approval — the clean final answer")
    rejection_reason: Optional[str] = Field(None, description="High level reason for rejection")