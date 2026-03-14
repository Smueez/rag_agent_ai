from enum import Enum


class PlannerAgentActions(str, Enum):
    retrieval = "retrieval"
    convo = "convo"