from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    response_generation_attempt: int = Field(ge=1, description="Number of attempts to generate a response", default=1)
    grounding_threshold: float = Field(gt=0, le=1, description="Minimum value of grounding threshold", default=0.75)
    total_chunk: int = Field(gt=0, description="Total number of chunks", default=5)

class QueryResponseModel(BaseModel):
    query: str
    response: Optional[str] = None
    model_confidence: float = Field(gt=0, description="Confidence threshold", default=0)
    total_time_taken_in_sec: float = Field(ge=0, description= "Total time taken in seconds", default=0)

    @staticmethod
    def empty():
        return QueryResponseModel(
            query='',
            response='',
            model_confidence=0,
            total_time_taken_in_sec=0,
        )

