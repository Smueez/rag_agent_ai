from typing import Optional

from models.critic_agent_models import CriticResponseModel, CriticFeedback
from models.planner_agent_response_model import PlannerAgentResponseModel
from services.agent_services.agents.critic_agent_service import CriticAgentService
from loguru import logger

class CriticAgentsPipelinesService:
    @staticmethod
    async def critic_agent_pipeline(panner_agent_response: PlannerAgentResponseModel, attempt: int, grounding_threshold: float = 0.75) -> Optional[CriticResponseModel]:
        """
            Runs Re-Ranking ->
            Generator → Grounding → Critic loop with N retries.

            Args:
                panner_agent_response: retrieved information from planner agent
                attempt: maximum number of retry attempts (default 1)
                grounding_threshold: minimum grounding score to approve

            Returns:
                Final CriticResponseModel — approved or best attempt after N retries
            """
        logger.info(f"Running Critic Agent Pipeline for attempt {attempt} with grounding threshold {grounding_threshold}")

        feedback: Optional[CriticFeedback] = None

        best_attempt: Optional[CriticResponseModel] = None

        re_ranked_response = await CriticAgentService.re_ranking_agent(request=panner_agent_response)

        for att in range(1, attempt + 1):
            logger.info(f"Running Critic Agent Pipeline for attempt {att}")
            generator_aget_response = await CriticAgentService.response_generation_agent(request=re_ranked_response, feedback=feedback)

            grounding_response = await CriticAgentService.grounding_agent(request=generator_aget_response)

            critic_response = await CriticAgentService.critic_agent(request=grounding_response, attempt=att, grounding_threshold=grounding_threshold)
            best_attempt = critic_response
            logger.debug(f"Attempt's verdict {critic_response.verdict}")
            if critic_response.verdict == "approved":
                return critic_response
            logger.warning(f"Critic Agent Pipeline for attempt {att} not approved yet!")
            feedback = critic_response.feedback

        return best_attempt


