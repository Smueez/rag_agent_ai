import traceback

from fastapi import APIRouter, HTTPException

from loguru import logger

from config import Settings
from models.request_models import QueryRequest
from usecase.query_usecase import QueryUseCase

router = APIRouter()
settings = Settings()


@router.post("/query")
async def query(request: QueryRequest):
    """
    Query the agent without streaming (for testing)
    """
    try:
        return await QueryUseCase.call(request.query, attempt=request.response_generation_attempt, grounding_threshold=request.grounding_threshold)

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in query: {e}")
        raise HTTPException(status_code=500, detail=str(e))