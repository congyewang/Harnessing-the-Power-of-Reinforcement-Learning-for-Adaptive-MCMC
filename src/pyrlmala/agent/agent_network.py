from abc import ABC, abstractmethod
from typing import Dict, List, Union

import torch
import torch.nn as nn
from gymnasium.vector import SyncVectorEnv
from jaxtyping import Float


class AgentNetworkBase(ABC, nn.Module):
    def __init__(
        self, envs: SyncVectorEnv, config: Dict[str, Union[List[int], str]]
    ) -> None:
        super().__init__()

        self.envs = envs

        # Getting Input and Output Sizes
        input_size = self._get_input_size()
        output_size = self._get_output_size()

        # Getting Hidden Layer Information from the Configuration
        hidden_layers = config.get("hidden_layers", [8, 8])
        activation_function_name = config.get("activation_function", "ReLU")

        # Getting Activation Function
        activation_function = getattr(nn, activation_function_name)()

        # Defining the Layers
        layers = [nn.Linear(input_size, hidden_layers[0]), activation_function]

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
        raise NotImplementedError(
            "_get_input_size method must be implemented in the subclass."
        )

    def _get_output_size(self) -> int:
        return 1

    @abstractmethod
    def forward(self, *args, **kwargs) -> Float[torch.Tensor, "step_size or q_value"]:
        raise NotImplementedError("forward method must be implemented in the subclass.")
