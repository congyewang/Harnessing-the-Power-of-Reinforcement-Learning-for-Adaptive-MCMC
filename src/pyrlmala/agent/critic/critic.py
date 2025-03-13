import torch
from jaxtyping import Float

from ...config import QNetworkConfigParser
from ..agent_network import AgentNetworkBase


class QNetwork(AgentNetworkBase):
    """
    QNetwork is a class that represents the Q-network of the agent.

    Attributes:
        input_size (int): The input size of the network.
        config (QNetworkConfigParser): The configuration from the critic TOML file parsed by QNetworkConfigParser.
    """

    def __init__(self, input_size: int, config: QNetworkConfigParser):
        """
        Initialize the QNetwork.

        Args:
            input_size (int): The input size of the network.
            config (QNetworkConfigParser): The configuration from the critic TOML file parsed by QNetworkConfigParser.
        """
        super().__init__(input_size=input_size, config=config)

    def _get_input_size(self) -> int:
        """
        Get the input size of the network.

        Returns:
            int: The input size of the network.
        """
        return self.input_size

    def forward(
        self,
        observation: Float[torch.Tensor, "current_sample proposed_sample"],
        action: Float[torch.Tensor, "current_step_size proposed_step_size"],
    ) -> Float[torch.Tensor, "q_value"]:
        """
        Forward pass of the network.

        Returns:
            Float[torch.Tensor, "q_value"]: The Q-value of the network.
        """
        x = torch.cat([observation, action], 1)
        return self.network(x)
