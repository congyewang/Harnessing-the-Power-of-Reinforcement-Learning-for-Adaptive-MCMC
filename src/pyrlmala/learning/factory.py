from abc import ABC, abstractmethod
from typing import Callable, Type

from .learning import LearningDDPG, LearningInterface, LearningTD3
from .preparation import PreparationDDPG, PreparationTD3


class LearningAlgorithmFactory(ABC):
    """
    Factory class for creating learning algorithms based on reinforcement learning strategies.

    Attributes:
        hyperparameter_config_path (str): The path to the hyperparameter configuration file.
        actor_config_path (str): The path to the actor configuration file.
        critic_config_path (str): The path to the critic configuration file.
    """

    @abstractmethod
    def create(self, **kwargs):
        """
        Create a learning algorithm instance. This method should be implemented in the subclass.

        Raises:
            NotImplementedError: create method is not implemented.
        """
        raise NotImplementedError("create method is not implemented.")

    @staticmethod
    def register_algorithm(
        algorithm_name: str,
        hyperparameter_config_path: str,
        actor_config_path: str,
        critic_config_path: str,
    ) -> Callable[[Type["LearningAlgorithmFactory"]], Type["LearningAlgorithmFactory"]]:
        """
        Register the algorithm to the factory.

        Args:
            algorithm_name (str): The name of the algorithm.
            hyperparameter_config_path (str): The path to the hyperparameter configuration file.
            actor_config_path (str): The path to the actor configuration file.
            critic_config_path (str): The path to the critic configuration file.

        Returns:
            Callable[[Type[LearningAlgorithmFactory]], Type[LearningAlgorithmFactory]]: The decorator.
        """

        def decorator(cls) -> Type["LearningAlgorithmFactory"]:
            """
            Decorator function.

            Args:
                cls (Type[LearningAlgorithmFactory]): The class.

            Returns:
                Type[LearningAlgorithmFactory]: The class.
            """

            def create_instance(
                hyperparameter_config_path=hyperparameter_config_path,
                actor_config_path=actor_config_path,
                critic_config_path=critic_config_path,
                **kwargs,
            ) -> "LearningAlgorithmFactory":
                """
                Create an instance of the learning algorithm.

                Args:
                    hyperparameter_config_path (str, optional): The path to the hyperparameter configuration file. Defaults to hyperparameter_config_path.
                    actor_config_path (str, optional): The path to the actor configuration file. Defaults to actor_config_path.
                    critic_config_path (str, optional): The path to the critic configuration file. Defaults to critic_config_path.

                Returns:
                    LearningAlgorithmFactory: The learning algorithm instance.
                """
                return cls(
                    hyperparameter_config_path=hyperparameter_config_path,
                    actor_config_path=actor_config_path,
                    critic_config_path=critic_config_path,
                ).create(**kwargs)

            LearningFactory.register_factory(algorithm_name, create_instance)
            return cls

        return decorator


class LearningFactory:
    """
    Factory class for creating learning algorithms based on reinforcement learning strategies.

    Attributes:
        _factories (Dict[str, Callable[..., LearningAlgorithmFactory]]): The factories.
    """

    _factories = {}

    @classmethod
    def register_factory(
        cls, algorithm: str, factory_callable: Callable[..., "LearningAlgorithmFactory"]
    ) -> None:
        """
        Register the factory to the class.

        Args:
            algorithm (str): The algorithm name, which should be lower case. For example, "ddpg" and "td3".
            factory_callable (Callable[..., LearningAlgorithmFactory]): The factory callable.
        """
        cls._factories[algorithm.lower()] = factory_callable

    @classmethod
    def create_learning_instance(
        cls,
        algorithm: str,
        hyperparameter_config_path="",
        actor_config_path="",
        critic_config_path="",
        **kwargs,
    ) -> LearningInterface:
        """
        Create a learning algorithm instance.

        Args:
            algorithm (str): The algorithm name, which should be lower case. For example, "ddpg" and "td3".
            hyperparameter_config_path (str, optional): The path to the hyperparameter configuration file. Defaults to "".
            actor_config_path (str, optional): The path to the actor configuration file. Defaults to "".
            critic_config_path (str, optional): The path to the critic configuration file. Defaults to "".

        Raises:
            ValueError: Unsupported algorithm.

        Returns:
            LearningInterface: The learning algorithm instance.
        """
        factory_callable = cls._factories.get(algorithm.lower())
        if not factory_callable:
            raise ValueError(f"Unsupported algorithm: {algorithm}.")
        return factory_callable(
            hyperparameter_config_path=hyperparameter_config_path,
            actor_config_path=actor_config_path,
            critic_config_path=critic_config_path,
            **kwargs,
        )


@LearningAlgorithmFactory.register_algorithm(
    "ddpg",
    hyperparameter_config_path="config/ddpg.toml",
    actor_config_path="config/actor.toml",
    critic_config_path="config/critic.toml",
)
class DDPGFactory(LearningAlgorithmFactory):
    """
    Factory class for creating learning algorithms based on reinforcement learning strategies.

    Attributes:
        hyperparameter_config_path (str): The path to the hyperparameter configuration file.
        actor_config_path (str): The path to the actor configuration file.
        critic_config_path (str): The path to the critic configuration file.
    """

    def __init__(
        self,
        hyperparameter_config_path: str,
        actor_config_path: str,
        critic_config_path: str,
    ) -> None:
        """
        Factory class for creating learning algorithms based on reinforcement learning strategies.

        Args:
            hyperparameter_config_path (str): The path to the hyperparameter configuration file.
            actor_config_path (str): The path to the actor configuration file.
            critic_config_path (str): The path to the critic configuration file.
        """
        self.hyperparameter_config_path = hyperparameter_config_path
        self.actor_config_path = actor_config_path
        self.critic_config_path = critic_config_path

    def create(self, **kwargs) -> LearningDDPG:
        """
        Create a learning algorithm instance.

        Returns:
            LearningDDPG: The learning algorithm instance.
        """
        return PreparationDDPG(
            hyperparameter_config_path=self.hyperparameter_config_path,
            actor_config_path=self.actor_config_path,
            critic_config_path=self.critic_config_path,
            **kwargs,
        ).create()


@LearningAlgorithmFactory.register_algorithm(
    "td3",
    hyperparameter_config_path="config/td3.toml",
    actor_config_path="config/actor.toml",
    critic_config_path="config/critic.toml",
)
class TD3Factory(LearningAlgorithmFactory):
    """
    Factory class for creating learning algorithms based on reinforcement learning strategies.

    Attributes:
        hyperparameter_config_path (str): The path to the hyperparameter configuration file.
        actor_config_path (str): The path to the actor configuration file.
        critic_config_path (str): The path to the critic configuration file.
    """

    def __init__(
        self,
        hyperparameter_config_path: str,
        actor_config_path: str,
        critic_config_path: str,
    ) -> None:
        """
        Factory class for creating learning algorithms based on reinforcement learning strategies.

        Args:
            hyperparameter_config_path (str): The path to the hyperparameter configuration file.
            actor_config_path (str): The path to the actor configuration file.
            critic_config_path (str): The path to the critic configuration file.
        """
        self.hyperparameter_config_path = hyperparameter_config_path
        self.actor_config_path = actor_config_path
        self.critic_config_path = critic_config_path

    def create(self, **kwargs) -> LearningTD3:
        """
        Create a learning algorithm instance.

        Returns:
            LearningTD3: The learning algorithm instance.
        """
        return PreparationTD3(
            hyperparameter_config_path=self.hyperparameter_config_path,
            actor_config_path=self.actor_config_path,
            critic_config_path=self.critic_config_path,
            **kwargs,
        ).create()
