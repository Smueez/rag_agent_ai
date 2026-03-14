import json

from services.agent_services.agents.planner_agents_service import RetrivalAgentService
from loguru import logger

class QueryUseCase:
    @staticmethod
    async def call(query:str) -> dict:
        planner_agent_result = await RetrivalAgentService.query_planner_agent(query)
        return planner_agent_result.model_dump()