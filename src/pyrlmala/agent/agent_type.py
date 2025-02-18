from enum import Enum


class AgentType(Enum):
    """
    Enum for the agent type.

    Attributes:
        ACTOR (str): The actor agent.
        CRITIC (str): The critic agent.
    """

    ACTOR = "actor"
    CRITIC = "critic"
