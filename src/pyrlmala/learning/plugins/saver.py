import re

import torch

from ...agent import AgentType
from ...utils import Toolbox
from ..learning import LearningInterface
from .base import PluginBase


class SaverBase(PluginBase):
    def __init__(
        self,
        learning_instance: LearningInterface,
        folder_path: str,
        save_after_steps: int,
        save_frequency: int,
    ) -> None:
        """
        Save the model. The model is saved as a dictionary. The model is saved after a certain number of steps and at a certain frequency.

        Args:
            learning_instance (LearningInterface): The learning instance.
            folder_path (str): The path to save the model.
            save_after_steps (int): Steps after which to save the model.
            save_frequency (int): Frequency of saving the model.
        """
        super().__init__(learning_instance=learning_instance)
        self.folder_path = folder_path
        self.save_after_steps = save_after_steps
        self.save_frequency = save_frequency

    def _get_current_step(self) -> int:
        """
        Get the current step of the environment.

        Returns:
            int: The current step.
        """
        return self.learning_instance.env.envs[0].get_wrapper_attr("current_step")

    def _get_algorithm_name(self) -> str:
        """
        Get the algorithm name.

        Returns:
            str: The algorithm name.
        """
        # Get the class name of the learning instance
        class_name = type(self.learning_instance).__name__

        # Regular expression pattern
        pattern = r"Learning(.+)"

        # Extract the algorithm name
        match = re.search(pattern, class_name)

        if match:
            extracted_name = match.group(1)
            return extracted_name.lower()
        else:
            raise ValueError("The algorithm name is not found.")

    def save_model_dict(self, agent_type: AgentType) -> None:
        """
        Save the model dictionary.

        Args:
            agent_type (AgentType): The agent type.
        """
        algorithm_name = self._get_algorithm_name()
        current_step = self._get_current_step()

        if current_step % self.save_after_steps == 0 and current_step != 0:
            if current_step % self.save_frequency == 0:
                model_path = f"{self.folder_path}/{algorithm_name}_{agent_type.value}.step_{current_step}.pth"

                # Create the folder if it does not exist
                Toolbox.create_folder(model_path)

                # Save the model
                agent_state = getattr(
                    self.learning_instance, agent_type.value
                ).state_dict()
                torch.save(agent_state, model_path)


class ActorSaver(SaverBase):
    """
    Save the actor model. The actor model is saved as a dictionary.

    Attributes:
        learning_instance (LearningInterface): The learning instance.
        folder_path (str): The path to save the actor model.
        save_after_steps (int): Steps after which to save the actor model. Default is 1.
        save_frequency (int): Frequency of saving the actor model. Default is 1.
    """

    def __init__(
        self,
        learning_instance: LearningInterface,
        folder_path: str = "./weights/actor",
        save_after_steps: int = 1,
        save_frequency: int = 1,
    ) -> None:
        """
        Save the actor model. The actor model is saved as a dictionary.

        Args:
            learning_instance (LearningInterface): The learning instance.
            folder_path (str): The path to save the actor model.
            save_after_steps (int): Steps after which to save the actor model. Default is 1.
            save_frequency (int): Frequency of saving the actor model. Default is 1.
        """
        super().__init__(
            learning_instance, folder_path, save_after_steps, save_frequency
        )

    def execute(self) -> None:
        """
        Save the actor model. The actor model is saved as a dictionary.
        """
        self.save_model_dict(AgentType.ACTOR)


class CriticSaver(SaverBase):
    """
    Save the critic model. The critic model is saved as a dictionary.

    Attributes:
        learning_instance (LearningInterface): The learning instance.
        folder_path (str): The path to save the critic model.
        save_after_steps (int): Steps after which to save the critic model. Default is 1.
        save_frequency (int): Frequency of saving the critic model. Default is 1.
    """

    def __init__(
        self,
        learning_instance: LearningInterface,
        folder_path: str = "./weights/critic",
        save_after_steps: int = 1,
        save_frequency: int = 1,
    ) -> None:
        """
        Save the critic model. The critic model is saved as a dictionary.

        Args:
            learning_instance (LearningInterface): The learning instance.
            folder_path (str): The path to save the critic model.
            save_after_steps (int): Steps after which to save the critic model. Default is 1.
            save_frequency (int): Frequency of saving the critic model. Default is 1.
        """
        super().__init__(
            learning_instance, folder_path, save_after_steps, save_frequency
        )

    def execute(self) -> None:
        """
        Save the critic model. The critic model is saved as a dictionary.
        """
        self.save_model_dict(AgentType.CRITIC)
