import json

from agents import Agent, ModelSettings, Runner

from models.critic_agent_models import ReRankingResponseModel, ReRankingInputModel
from models.planner_agent_response_model import PlannerAgentResponseModel
from utils.file_utils import read_agent_instruction_file

class CriticAgentService:
    """Agent Service for critic agents."""

    @staticmethod
    async def re_ranking_agent(request: PlannerAgentResponseModel)->ReRankingResponseModel:
        instructions = read_agent_instruction_file("re_ranked_agent")
        re_ranking_agent = Agent(
            name="Re-Ranked Agent",
            instructions=instructions,
            model="gpt-40-mini",
            model_settings=ModelSettings(temperature=0.1),
            output_type=ReRankingResponseModel,
        )
        input_payload = ReRankingInputModel(
            query=request.rewritten_query,
            chunks=request.top_k_chunks
        )
        result = await Runner.run(re_ranking_agent, input=json.dumps(input_payload))
        return result.final_output

    @staticmethod
    async def response_generation(request: ReRankingResponseModel):
        pass
