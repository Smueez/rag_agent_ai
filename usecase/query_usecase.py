from services.agent_services.agents.planner_agents_service import RetrivalAgentService

from services.agent_services.critic_agents_pipelines_service import CriticAgentsPipelinesService


class QueryUseCase:
    @staticmethod
    async def call(query:str, attempt: int, grounding_threshold: float) -> dict:
        planner_agent_result = await RetrivalAgentService.query_planner_agent(query)
        final_response = await CriticAgentsPipelinesService.critic_agent_pipeline(
            panner_agent_response=planner_agent_result,
            attempt=attempt,
            grounding_threshold=grounding_threshold
        )
        return final_response.model_dump()