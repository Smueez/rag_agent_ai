import json
from typing import Optional

from agents import Agent, ModelSettings, Runner

from models.critic_agent_models import ReRankingResponseModel, ReRankingInputModel, ResponseGeneratorModel, \
    ReturnedGeneratorModel, GroundingResponseModel, CriticFeedback, CriticResponseModel
from models.planner_agent_response_model import PlannerAgentResponseModel
from utils.file_utils import read_agent_instruction_file
from loguru import logger


class CriticAgentService:
    """Agent Service for critic agents."""

    @staticmethod
    async def re_ranking_agent(request: PlannerAgentResponseModel)->ReRankingResponseModel:
        logger.info(f"Re-ranking agent running start")
        instructions = read_agent_instruction_file("re_ranked_agent")
        re_ranking_agent = Agent(
            name="Re-Ranked Agent",
            instructions=instructions,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.1),
            output_type=ReRankingResponseModel,
        )
        input_payload = ReRankingInputModel(
            query=request.rewritten_query,
            chunks=request.top_k_chunks
        )
        result = await Runner.run(re_ranking_agent, input=json.dumps(input_payload.model_dump()))
        return result.final_output

    @staticmethod
    async def response_generation_agent(request: ReRankingResponseModel, feedback: Optional[CriticFeedback] = None)->ReturnedGeneratorModel:
        logger.info(f"Response generation agent running start")
        instructions = read_agent_instruction_file("response_generation_agent")
        generation_agent = Agent(
            name="Response Generation Agent",
            instructions=instructions,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.3),
            output_type=ResponseGeneratorModel,
        )

        input_payload = {
            "re_ranking_response": request.model_dump(),
        }
        if feedback:
            input_payload["feedback_from_critic"] = feedback.model_dump()
        result = await Runner.run(generation_agent, input=json.dumps(input_payload))
        return ReturnedGeneratorModel(
            query=result.final_output.query,
            chunks=request.ranked_chunks,
            response=result.final_output.response
        )

    @staticmethod
    async def grounding_agent(request: ReturnedGeneratorModel) -> GroundingResponseModel:
        logger.info(f"Grounding agent running start")
        instructions = read_agent_instruction_file("grounding_agent")
        grounding_ai_agent = Agent(
            name="Grounding Agent",
            instructions=instructions,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.1),
            output_type=GroundingResponseModel,
        )

        result = await Runner.run(grounding_ai_agent, input=json.dumps(request.model_dump()))
        return result.final_output

    @staticmethod
    async def critic_agent(request: GroundingResponseModel, attempt: int, grounding_threshold: float = 0.75) -> CriticResponseModel:
        logger.info(f"Critic agent running start")
        instructions = read_agent_instruction_file("critic_agent")
        critic_ai_agent = Agent(
            name="Critic Agent",
            instructions=instructions,
            model="gpt-4o-mini",
            model_settings=ModelSettings(temperature=0.1),
            output_type=CriticResponseModel,
        )
        input_payload = request.model_dump()
        input_payload["grounding_threshold"] = grounding_threshold
        input_payload["attempt"] = attempt
        result = await Runner.run(critic_ai_agent, input=json.dumps(input_payload))
        return result.final_output