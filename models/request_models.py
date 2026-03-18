from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    response_generation_attempt: int = Field(ge=1, description="Number of attempts to generate a response", default=1)
    grounding_threshold: float = Field(gt=0, le=1, description="Minimum value of grounding threshold", default=0.75)
    total_chunk: int = Field(gt=0, description="Total number of chunks", default=5)


