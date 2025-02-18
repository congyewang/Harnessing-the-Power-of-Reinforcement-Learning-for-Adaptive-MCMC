from abc import ABC, abstractmethod
from typing import Optional

from ..learning import LearningInterface


class PluginBase(ABC):
    """
    Base class for all plugins.

    Attributes:
        learning (Optional[LearningInterface]): The learning interface object.
    """

    def __init__(self, learning_instance: Optional[LearningInterface]) -> None:
        """
        Constructor for PluginBase.

        Args:
            learning_instance (Optional[LearningInterface]): The learning interface object.
        """
        self.learning_instance = learning_instance

    @abstractmethod
    def execute(self) -> None:
        """
        Execute the plugin.

        Raises:
            NotImplementedError: Indicates that the method has not been implemented in a subclass.
        """
        raise NotImplementedError("Method not implemented")
