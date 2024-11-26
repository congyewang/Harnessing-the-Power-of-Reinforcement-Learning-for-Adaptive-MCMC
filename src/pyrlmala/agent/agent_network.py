from abc import ABC, abstractmethod
from typing import Callable, List

import torch
import torch.nn as nn
from gymnasium.vector import SyncVectorEnv
from jaxtyping import Float

from ..config import PolicyNetworkConfigParser, QNetworkConfigParser


class AgentNetworkBase(ABC, nn.Module):
    """
    Base Class for the Agent Network.

    Attributes:
        envs (SyncVectorEnv): The SyncVectorEnv environment inherited from env.MCMCEnvBase Class.
        network (nn.Module): The Neural Network.
    """

    def __init__(
        self,
        envs: SyncVectorEnv,
        config: PolicyNetworkConfigParser | QNetworkConfigParser,
    ) -> None:
        """
        Initializes the Agent Network.

        Args:
            envs (SyncVectorEnv): The SyncVectorEnv environment inherited from env.MCMCEnvBase Class.
            config (PolicyNetworkConfigParser | QNetworkConfigParser): The Configuration.
        """
        super().__init__()

        self.envs = envs

        # Getting Input and Output Sizes
        input_size = self._get_input_size()
        output_size = self._get_output_size()

        # Getting Hidden Layer Information from the Configuration
        hidden_layers = getattr(config, "hidden_layers", [8, 8])
        activation_function_name = getattr(config, "activation_function", "ReLU")

        # Getting Activation Function
        activation_function = getattr(nn, activation_function_name)()

        # Defining the Layers
        layers: List[
            nn.Module
            | Callable[[Float[torch.Tensor, "input"]], Float[torch.Tensor, "output"]]
        ] = [nn.Linear(input_size, hidden_layers[0]), activation_function]

        # Defining Hidden Layers Dynamically
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(activation_function)

        # Defining the Output Layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Using Sequential to Define the Network
        self.network = nn.Sequential(*layers)

    @abstractmethod
    def _get_input_size(self) -> int:
        """
        Gets the Input Size of the Network.

        Raises:
            NotImplementedError: If the Method is not Implemented.

        Returns:
            int: The Input Size.
        """
        raise NotImplementedError(
            "_get_input_size method must be implemented in the subclass."
        )

    def _get_output_size(self) -> int:
        """
        Gets the Output Size of the Network.

        Returns:
            int: The Output Size.
        """
        return 1

    @abstractmethod
    def forward(self, *args, **kwargs) -> Float[torch.Tensor, "step_size or q_value"]:
        """
        Forward Method of the Network.

        Raises:
            NotImplementedError: If the Method is not Implemented.
        """
        raise NotImplementedError("forward method must be implemented in the subclass.")
