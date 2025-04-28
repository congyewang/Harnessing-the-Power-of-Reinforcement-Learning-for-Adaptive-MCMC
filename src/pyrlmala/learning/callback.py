import threading
from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger

from .events import TrainEvents
from .learning import LearningInterface
from .observer import ConfigObserver
from .plugins import (
    ActorLearningRateConfig,
    ActorLearningRateSlider,
    ActorSaver,
    CriticLearningRateConfig,
    CriticSaver,
    TrainingVisualizer,
)


class CallbackBase(ABC):
    """
    Base class for callbacks.

    Attributes:
        learning_instance (LearningInterface): The learning instance to be trained.
    """

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
    def _register_all(self, *args: Any, **kwargs: Any) -> None:
        """
        Register all the events.
        """
        raise NotImplementedError("Method `_register_all` must be implemented.")

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
    """
    Callback for training the learning instance.

    Attributes:
        learning_instance (LearningInterface): The learning instance to be trained.
        plotter (TrainingVisualizer): The training visualizer.
        observer (ConfigObserver): The configuration observer.
        runtime_actor_learning_rate_slider (ActorLearningRateSlider): The actor learning rate slider.
    """

    def __init__(
        self,
        learning_instance: LearningInterface,
        plot_frequency: int = 10,
        num_of_mesh: int = 10,
        auto_start: bool = True,
        runtime_config_path: Optional[str] = None,
        actor_folder_path: str = "./weights/actor",
        actor_save_after_steps: int = 1,
        actor_save_frequency: int = 1,
        critic_folder_path: str = "./weights/critic",
        critic_save_after_steps: int = 1,
        critic_save_frequency: int = 1,
    ) -> None:
        """
        Callback for plotting the policy of the learning instance in 2D. Only works for 2D environments.

        Args:
            learning_instance (LearningInterface): The learning instance to be trained.
            plot_frequency (int, optional): The frequency of plotting the policy. Defaults to 10.
            num_of_mesh (int, optional): The number of mesh points. Defaults to 10.
            auto_start (bool, optional): Whether to start the observer automatically. Defaults to True.
            runtime_config_path (Optional[str], optional): The path to the runtime configuration file. Defaults to None.
            actor_folder_path (str, optional): The path to save the actor model. Defaults to "./weights/actor".
            actor_save_after_steps (int, optional): Steps after which to save the actor model. Defaults to 1.
            actor_save_frequency (int, optional): Frequency of saving the actor model. Defaults to 1.
            critic_folder_path (str, optional): The path to save the critic model. Defaults to "./weights/critic".
            critic_save_after_steps (int, optional): Steps after which to save the critic model. Defaults to 1.
            critic_save_frequency (int, optional): Frequency of saving the critic model. Defaults to 1.
        """
        super().__init__(learning_instance)

        self.plotter = TrainingVisualizer(
            learning_instance, plot_frequency, num_of_mesh
        )

        self.actor_saver = ActorSaver(
            learning_instance,
            actor_folder_path,
            actor_save_after_steps,
            actor_save_frequency,
        )
        self.critic_saver = CriticSaver(
            learning_instance,
            critic_folder_path,
            critic_save_after_steps,
            critic_save_frequency,
        )

        if auto_start:
            if runtime_config_path is None:
                raise ValueError("Runtime config path must be provided.")

            self._observer_start(learning_instance, runtime_config_path)

        self._register_all(learning_instance)

    def __del__(self) -> None:
        """
        Ensure observer is stopped when the instance is deleted.
        """
        try:
            self.observer.stop()
            logger.info(f"{self.__class__.__name__} observer stopped.")
        except Exception as e:
            logger.error(f"Error stopping observer: {e}")
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
        logger.info(f"{self.__class__.__name__} observer started.")

        self.actor_learning_rate_slider = ActorLearningRateSlider(runtime_config_path)
        # self._make_actor_learning_rate_slider()

    def _execute_runtime_config(self) -> None:
        """
        Execute the runtime configuration.
        """
        self.runtime_actor_learning_rate_configger.execute()

    def _make_actor_learning_rate_slider(self) -> None:
        """
        Make the actor learning rate slider.
        """
        self.actor_learning_rate_slider.execute()

    def _save_actor(self) -> None:
        """
        Save the actor.
        """
        self.actor_saver.execute()

    def _save_critic(self) -> None:
        """
        Save the critic.
        """
        self.critic_saver.execute()

    def _callback(self) -> None:
        """
        Execute the callback.
        """
        threading.Thread(target=self.plotter.execute).start()
        self._save_actor()
        # self._save_critic()

    def _register_all(self, learning_instance: LearningInterface) -> None:
        learning_instance.event_manager.register(
            TrainEvents.WITHIN_TRAIN, self._callback
        )
