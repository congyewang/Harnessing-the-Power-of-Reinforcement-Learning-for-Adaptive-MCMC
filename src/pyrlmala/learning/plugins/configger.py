from enum import Enum
from typing import Dict

import tomllib

from ..learning import LearningInterface
from .base import PluginBase


class AgentType(Enum):
    """
    Enum for the agent type.

    Attributes:
        ACTOR (str): The actor agent.
        CRITIC (str): The critic agent.
    """

    ACTOR = "actor"
    CRITIC = "critic"


class RuntimeCongfigBase(PluginBase):
    """
    Base class for runtime configuration plugins.

    Attributes:
        learning_instance (LearningInterface): The learning instance.
        file_path (str): The path to the configuration file.
    """

    def __init__(self, learning_instance: LearningInterface, file_path: str):
        """
        Base class for runtime configuration plugins.

        Args:
            learning_instance (LearningInterface): The learning instance.
            file_path (str): The path to the configuration file.
        """
        self.learning_instance = learning_instance
        self.file_path = file_path

    def _get_config(self, file_path: str) -> Dict[str, float | int | bool]:
        """
        Get the configuration from the file.

        Args:
            file_path (str): The path to the configuration file.

        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If there is an error decoding the file.

        Returns:
            Dict[str, float | int | bool]: The configuration dictionary.
        """
        try:
            with open(file_path, "rb") as f:
                config_dict = tomllib.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{file_path}' not found.")
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Error decoding TOML file: {e}")

        return config_dict


class LearningRateConfigBase(RuntimeCongfigBase):
    """
    Base class for learning rate configuration plugins.

    Attributes:
        learning_instance (LearningInterface): The learning instance.
        file_path (str): The path to the configuration file.
    """

    def _check_learning_rate(self, learning_rate: float | None) -> None:
        """
        Check if the learning rate is valid.

        Args:
            learning_rate (float | None): The learning rate.

        Raises:
            ValueError: If the learning rate is not a number or is less than or equal to zero.
        """
        if not isinstance(learning_rate, (int, float)):
            raise ValueError("Learning rate must be a number.")

        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than zero.")

    def _get_runtime_learning_rate(
        self,
        learning_rate_type: AgentType,
        default_learning_rate: float,
        config_dict: Dict[str, float | int | bool],
    ) -> float:
        """
        Get the runtime learning rate.

        Args:
            learning_rate_type (AgentType): The agent type.
            default_learning_rate (float): The default learning rate.
            config_dict (Dict[str, float  |  int  |  bool]): The configuration dictionary.

        Returns:
            float: The runtime learning rate.
        """
        runtime_learning_rate = config_dict.get(
            f"{learning_rate_type.value}_learning_rate", default_learning_rate
        )

        return runtime_learning_rate

    def _change_learning_rate(
        self, learning_rate_type: AgentType, config_dict: Dict[str, float | int | bool]
    ) -> None:
        """
        Change the learning rate.

        Args:
            learning_rate_type (AgentType): The agent type.
            config_dict (Dict[str, float  |  int  |  bool]): The configuration dictionary.
        """
        optimizer = getattr(
            self.learning_instance, f"{learning_rate_type.value}_optimizer"
        )
        default_learning_rate = optimizer.param_groups[0].get("lr", None)
        self._check_learning_rate(default_learning_rate)

        runtime_learning_rate = self._get_runtime_learning_rate(
            learning_rate_type, default_learning_rate, config_dict
        )
        self._check_learning_rate(runtime_learning_rate)
        optimizer.param_groups[0].update(lr=runtime_learning_rate)


class ActorLearningRateConfig(LearningRateConfigBase):
    """
    Change the actor learning rate.

    Attributes:
        learning_instance (LearningInterface): The learning instance.
        file_path (str): The path to the configuration file.
    """

    def execute(self) -> None:
        """
        Execute the plugin.
        """
        self._change_learning_rate(AgentType.ACTOR, self._get_config(self.file_path))


class CriticLearningRateConfig(LearningRateConfigBase):
    """
    Change the critic learning rate.

    Attributes:
        learning_instance (LearningInterface): The learning instance.
        file_path (str): The path to the configuration file.
    """

    def execute(self) -> None:
        """
        Execute the plugin.
        """
        self._change_learning_rate(AgentType.CRITIC, self._get_config(self.file_path))
