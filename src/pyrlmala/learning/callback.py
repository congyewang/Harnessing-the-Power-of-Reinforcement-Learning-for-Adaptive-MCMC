from abc import ABC, abstractmethod
from typing import Optional

from .learning import LearningInterface
from .observer import ConfigObserver
from .plugins import (
    ActorLearningRateConfig,
    CriticLearningRateConfig,
    TrainingVisualizer,
)


class CallbackBase(ABC):
    def __init__(
        self,
        learning_instance: LearningInterface,
    ) -> None:
        """
        Base class for callbacks.

        Args:
            learning_instance (LearningInterface): The learning instance to be trained.
        """
        self.learning_instance = learning_instance
        self.learning_instance.callback = self._callback

    @abstractmethod
    def _callback(self) -> None:
        """
        Execute the callback.
        """
        raise NotImplementedError("Method `execute` must be implemented.")

    def train(self) -> None:
        """
        Train the Learning Instance.
        """
        self.learning_instance.train()


class Callback(CallbackBase):
    def __init__(
        self,
        learning_instance: LearningInterface,
        plot_frequency: int = 10,
        num_of_mesh: int = 10,
        auto_start: bool = True,
        runtime_config_path: Optional[str] = None,
    ) -> None:
        """
        Callback for plotting the policy of the learning instance in 2D. Only works for 2D environments.

        Args:
            learning_instance (LearningInterface): The learning instance to be trained.
            ranges (Tuple[Tuple[int, int, float], Tuple[int, int, float]]): The ranges for the 2D plot.
            plot_frequency (int, optional): The frequency of plotting the policy. Defaults to 10.
        """
        super().__init__(learning_instance)

        self.plotter = TrainingVisualizer(
            learning_instance, plot_frequency, num_of_mesh
        )

        if auto_start:
            if runtime_config_path is None:
                raise ValueError("Runtime config path must be provided.")

            self._observer_start(learning_instance, runtime_config_path)

    def __del__(self) -> None:
        """
        Ensure observer is stopped when the instance is deleted.
        """
        try:
            self.observer.stop()
            print(f"{self.__class__.__name__} observer stopped.")
        except Exception as e:
            print(f"Error stopping observer: {e}")
        finally:
            if hasattr(super(), "__del__"):
                super().__del__()

    def _observer_start(
        self, learning_instance: LearningInterface, runtime_config_path: str
    ) -> None:
        """
        Start the observer for runtime configuration.

        Args:
            learning_instance (LearningInterface): The learning instance to be trained.
            runtime_config_path (str): The path to the runtime configuration file.
        """
        self.runtime_actor_learning_rate_configger = ActorLearningRateConfig(
            learning_instance, runtime_config_path
        )
        self.runtime_critic_learning_rate_configger = CriticLearningRateConfig(
            learning_instance, runtime_config_path
        )

        self.observer = ConfigObserver(
            runtime_config_path,
            self._execute_runtime_config,
        )

        self.observer.start()
        print(f"{self.__class__.__name__} observer started.")

    def _execute_runtime_config(self) -> None:
        """
        Execute the runtime configuration.
        """
        self.runtime_actor_learning_rate_configger.execute()

    def _callback(self) -> None:
        """
        Execute the callback.
        """
        self.plotter.execute()
