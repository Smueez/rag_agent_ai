import traceback

from fastapi import APIRouter, HTTPException

from loguru import logger

from config import Settings
from models.request_response_models import QueryRequest, QueryResponseModel
from usecase.query_usecase import QueryUseCase

router = APIRouter()
settings = Settings()


@router.post("/query", response_model=QueryResponseModel)
async def query(request: QueryRequest):
    """
    Query the agent without streaming
    """
    try:
        result = await QueryUseCase.call(request.query, attempt=request.response_generation_attempt, grounding_threshold=request.grounding_threshold)
        return result.model_dump()

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in query: {e}")
        raise HTTPException(status_code=500, detail=str(e))