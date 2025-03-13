import torch
from jaxtyping import Float

from ...config import PolicyNetworkConfigParser
from ..agent_network import AgentNetworkBase


class PolicyNetwork(AgentNetworkBase):
    """
    PolicyNetwork.

    Attributes:
        input_size (int): The input size of the network.
        config (PolicyNetworkConfigParser): The configuration from the actor TOML file parsed by PolicyNetworkConfigParser.
    """

    def __init__(self, input_size: int, config: PolicyNetworkConfigParser) -> None:
        """
        Initialize the PolicyNetwork.

        Args:
            input_size (int): The input size of the network.
            config (PolicyNetworkConfigParser): The configuration from the actor TOML file parsed by PolicyNetworkConfigParser.
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
        self, observation: Float[torch.Tensor, "current_sample proposed_sample"]
    ) -> Float[torch.Tensor, "current_step_size proposed_step_size"]:
        """
        Forward pass of the network.

        Returns:
            Float[torch.Tensor, "current_step_size proposed_step_size"]: The action of the network.
        """
        current_sample, proposed_sample = torch.tensor_split(observation, 2, dim=1)

        current_phi = self.network(current_sample)
        proposed_phi = self.network(proposed_sample)

        action = torch.concatenate([current_phi, proposed_phi], dim=1)

        return action
