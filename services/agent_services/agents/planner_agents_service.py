from agents import Agent, Runner, function_tool
from ..tools.tool import Tool
from models.planner_agent_response_model import PlannerAgentResponseModel
from utils.file_utils import read_agent_instruction_file


class RetrivalAgentService:


    @staticmethod
    async def query_planner_agent(query: str)->PlannerAgentResponseModel:
        instructions = read_agent_instruction_file("planner_agent")
        planner_agent = Agent(
            name="Planner Agent",
            instructions=instructions,
            tools=[RetrivalAgentService.query_classify_agent, RetrivalAgentService.query_rewrite_agent, Tool.fetch_data],
            model="gpt-4o-mini",
            output_type=PlannerAgentResponseModel
        )
        result = await Runner.run(planner_agent, input=f"User's query {query}")
        return result.final_output



    @staticmethod
    @function_tool
    async def query_classify_agent(query: str):
        """
        Classify whether the user's query requires knowledge base retrieval
        or is just a casual conversation.

        Args:
            query: the user's original message

        Returns:
            Either 'retrieval' or 'convo'
        """
        instructions = read_agent_instruction_file("classify_agent")

        classify_agent = Agent(name="Classify Agent", instructions=instructions, model="gpt-4o-mini", )
        result = await Runner.run(classify_agent, input=f"User's query {query}")
        return result.final_output



    @staticmethod
    @function_tool
    async def query_rewrite_agent(query: str):
        instructions = read_agent_instruction_file("rewrite_agent")
        rewrite_agent = Agent(
            name="Rewrite_agent Agent",
            instructions=instructions,
            model="gpt-4o-mini",
            handoff_description="Rewrite user's query if query with proper context."
        )
        result = await Runner.run(rewrite_agent, input=f"User's query {query}")
        return result.final_output