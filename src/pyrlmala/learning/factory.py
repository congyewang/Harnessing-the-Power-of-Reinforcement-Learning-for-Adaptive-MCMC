import json
import random
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import bridgestan as bs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
import torch.optim as optim
from gymnasium.envs.registration import EnvSpec
from posteriordb import PosteriorDatabase
from stable_baselines3.common.buffers import ReplayBuffer

from ..agent import PolicyNetwork, QNetwork
from ..config import (
    HyperparameterConfigParser,
    PolicyNetworkConfigParser,
    QNetworkConfigParser,
)
from ..envs import MCMCEnvBase
from . import LearningDDPG, LearningTD3


class PosteriorDBFunctionsGenerator:
    def __init__(
        self,
        model_name: str,
        posteriordb_path: str,
        posterior_data: Optional[Dict[str, float | int | List[float | int]]] = None,
    ) -> None:
        """
        Target functions generator for PosteriorDB. This class is used to generate target functions for a given model in PosteriorDB. The target functions are the log probability density function and its gradient.

        Args:
            model_name (str): The name of the model in PosteriorDB.
            posteriordb_path (str): The path to the PosteriorDB database.
            posterior_data (Dict[str, float  |  int  |  List[float  |  int]], optional): The parameter of the model. Defaults to None.
        """
        self.model_name = model_name
        self.posteriordb_path = posteriordb_path
        self.posterior_data = posterior_data

        self.stan_model = self.build_model()
        self.log_pdf = self.make_log_pdf()
        self.grad_log_pdf = self.make_grad_log_pdf()

    def build_model(self) -> bs.StanModel:
        """
        Builds a Stan model from the given Stan code and data.

        Returns:
            bs.StanModel: The Stan model.
        """
        pdb = PosteriorDatabase(self.posteriordb_path)

        posterior = pdb.posterior(self.model_name)
        stan_code = posterior.model.stan_code_file_path()

        if self.posterior_data is None:
            stan_data = json.dumps(posterior.data.values())
            print(stan_data)
        else:
            stan_data = json.dumps(self.posterior_data)

        model = bs.StanModel.from_stan_file(stan_code, stan_data)

        return model

    def make_log_pdf(
        self,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """
        Creates a log probability density function for the given posterior.

        Returns:
            Callable[[npt.ArrayLike], npt.ArrayLike]: A callable that computes the log density for given input.
        """
        return self.stan_model.log_density

    def make_grad_log_pdf(
        self,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """
        Creates a gradient of the log probability density function for the given posterior.

        Returns:
            Callable[[npt.ArrayLike], npt.ArrayLike]: A callable that computes the gradient of the log density for given input.
        """
        return lambda x: self.stan_model.log_density_gradient(x)[1]


class PreparationInterface(ABC):
    def __init__(
        self,
        initial_sample: npt.NDArray[np.float64],
        initial_covariance: Optional[npt.NDArray[np.float64]] = None,
        initial_step_size: npt.NDArray[np.float64] = np.array([1.0]),
        log_mode: bool = True,
        log_target_pdf: Optional[
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        ] = None,
        grad_log_target_pdf: Optional[
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        ] = None,
        model_name: Optional[str] = None,
        posteriordb_path: Optional[str] = None,
        posterior_data: Optional[Dict[str, float | int | List[float | int]]] = None,
        hyperparameter_config_path: str = "",
        actor_config_path: str = "",
        critic_config_path: str = "",
        compile: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Factory class for creating learning algorithms based on reinforcement learning strategies.

        Args:
            log_target_pdf (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], optional): Log probability density function of the target distribution. If not provided, it will be generated.
            grad_log_target_pdf (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], optional): Gradient of the log probability density function. If not provided, it will be generated.
            model_name (str, optional): Model name in PosteriorDB. Defaults to None.
            posteriordb_path (str, optional): The path of PosteriorDB. Defaults to None.
            posterior_data (Dict[str, float  |  int  |  List[float  |  int]], optional): The posterior data. Defaults to None.

        Raises:
            ValueError: If log_target_pdf or grad_log_target_pdf is not provided, model_name and posteriordb_path cannot be None.
        """
        # Initial Essential Attribution of Environment
        self.initial_sample = initial_sample
        if initial_covariance is None:
            self.initial_covariance = np.eye(len(initial_sample.flatten()))
        self.initial_step_size = initial_step_size
        self.log_mode = log_mode
        self.verbose = verbose

        # Make log target pdf and grad log target pdf
        self.log_target_pdf, self.grad_log_target_pdf = self.make_target_functions(
            log_target_pdf,
            grad_log_target_pdf,
            model_name,
            posteriordb_path,
            posterior_data,
        )

        # Make Args
        self.args, self.actor_config, self.critic_config = self.make_args(
            hyperparameter_config_path, actor_config_path, critic_config_path
        )

        # Set device
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and self.args.experiments.cuda
            else "cpu"
        )

        # Fixed random seed
        self.fixed_random_seed()

        # Make envs
        self.envs, self.predicted_envs = self.make_env()

        # Make Actor
        self.actor, self.target_actor = self.make_actor(compile)

        # Make Replay Buffer
        self.replay_buffer = self.make_replay_buffer()

    def make_target_functions(
        self,
        log_target_pdf: Optional[
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        ],
        grad_log_target_pdf: Optional[
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        ],
        model_name: Optional[str],
        posteriordb_path: Optional[str],
        posterior_data: Optional[Dict[str, float | int | List[float | int]]],
    ) -> Tuple[
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ]:
        """
        Creates target functions for the given model.

        Args:
            log_target_pdf (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): The log probability density function.
            grad_log_target_pdf (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): The gradient of the log probability density function.
            model_name (str): The name of the model in PosteriorDB.
            posteriordb_path (str): The path to the PosteriorDB database.
            posterior_data (Dict[str, float  |  int  |  List[float  |  int]]): The parameter of the model.

        Returns:
            Tuple[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]: The log probability density function and its gradient.

        Raises:
            ValueError: log_target_pdf and grad_log_target_pdf must be callable.
        """
        if log_target_pdf is None or grad_log_target_pdf is None:
            if model_name is None or posteriordb_path is None:
                raise ValueError(
                    "If log_target_pdf or grad_log_target_pdf is not provided, model_name and posteriordb_path cannot be None."
                )

            pdb_generator = PosteriorDBFunctionsGenerator(
                model_name=model_name,
                posteriordb_path=posteriordb_path,
                posterior_data=posterior_data,
            )
            log_target_pdf = pdb_generator.make_log_pdf()
            grad_log_target_pdf = pdb_generator.make_grad_log_pdf()

            return log_target_pdf, grad_log_target_pdf
        elif callable(log_target_pdf) and callable(grad_log_target_pdf):
            return log_target_pdf, grad_log_target_pdf
        else:
            raise ValueError("log_target_pdf and grad_log_target_pdf must be callable.")

    def make_args(
        self,
        hyperparameter_config_path: str,
        actor_config_path: str,
        critic_config_path: str,
    ) -> Tuple[
        HyperparameterConfigParser, PolicyNetworkConfigParser, QNetworkConfigParser
    ]:
        """
        Make Arguments.

        Args:
            hyperparameter_config_path (str): _description_
            actor_config_path (str): _description_
            critic_config_path (str): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            Tuple[HyperparameterConfigParser, PolicyNetworkConfigParser, QNetworkConfigParser]: _description_
        """
        if hyperparameter_config_path:
            args = HyperparameterConfigParser(hyperparameter_config_path)
        else:
            raise ValueError("hyperparameter_config_path must be provided.")

        if actor_config_path:
            actor_config = PolicyNetworkConfigParser(actor_config_path)
        else:
            raise ValueError("actor_config_path must be provided.")

        if critic_config_path:
            critic_config = QNetworkConfigParser(critic_config_path)
        else:
            raise ValueError("critic_config_path must be provided.")

        return args, actor_config, critic_config

    def fixed_random_seed(self) -> None:
        """
        Fixed random seed.

        Raises:
            ValueError: Seed must be an integer.
        """
        if not isinstance(self.args.experiments.seed, int):
            raise ValueError(
                f"Seed must be an integer. Got {type(self.args.experiments.seed)}."
            )

        random.seed(self.args.experiments.seed)
        np.random.seed(self.args.experiments.seed)
        torch.manual_seed(self.args.experiments.seed)
        torch.backends.cudnn.deterministic = self.args.experiments.torch_deterministic

    def init_env(
        self,
        env_id: str | EnvSpec,
        total_timesteps: int,
    ) -> Callable[[], MCMCEnvBase]:
        """
        Initialize environment to the function.

        Args:
            env_id (str | EnvSpec): The environment id.
        """

        def thunk():
            env = gym.make(
                id=env_id,
                log_target_pdf_unsafe=self.log_target_pdf,
                grad_log_target_pdf_unsafe=self.grad_log_target_pdf,
                initial_sample=self.initial_sample,
                initial_covariance=self.initial_covariance,
                initial_step_size=self.initial_step_size,
                total_timesteps=total_timesteps,
                log_mode=self.log_mode,
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(self.args.experiments.seed)

            return env

        return thunk

    def make_env(self) -> Tuple[gym.vector.SyncVectorEnv, gym.vector.SyncVectorEnv]:
        """
        Make environment and predict environment.

        Returns:
            Tuple[gym.vector.SyncVectorEnv, gym.vector.SyncVectorEnv]: The environment and predict environment.
        """
        envs = gym.vector.SyncVectorEnv(
            [
                self.init_env(
                    env_id=self.args.algorithm.general.env_id,
                    total_timesteps=self.args.algorithm.general.total_timesteps,
                )
            ]
        )
        assert isinstance(
            envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"
        envs.single_observation_space.dtype = np.float64

        predicted_envs = gym.vector.SyncVectorEnv(
            [
                self.init_env(
                    env_id=self.args.algorithm.general.env_id,
                    total_timesteps=self.args.algorithm.general.predicted_timesteps,
                )
            ]
        )
        assert isinstance(
            predicted_envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"
        predicted_envs.single_observation_space.dtype = np.float64

        return envs, predicted_envs

    def make_actor(self, compile: bool) -> Tuple[PolicyNetwork, PolicyNetwork]:
        """
        Make actor.

        Args:
            compile (bool): Whether to compile the model or not.

        Returns:
            PolicyNetwork: The actor.
        """
        actor = PolicyNetwork(self.envs, self.actor_config).to(self.device).double()
        target_actor = (
            PolicyNetwork(self.envs, self.actor_config).to(self.device).double()
        )
        if compile:
            actor = torch.compile(actor)
            target_actor = torch.compile(target_actor)

        return actor, target_actor

    @abstractmethod
    def make_critic(
        self, compile: bool
    ) -> Tuple[QNetwork, QNetwork] | Tuple[QNetwork, QNetwork, QNetwork, QNetwork]:
        raise NotImplementedError("make_critic method is not implemented.")

    def make_optimizer(
        self, actor: PolicyNetwork, critic: QNetwork
    ) -> Tuple[optim.Optimizer, optim.Optimizer]:
        """
        Make optimizer.

        Args:
            actor (PolicyNetwork): The actor.
            critic (QNetwork): The critic.

        Returns:
            optim.Optimizer: The optimizer.
        """
        actor_optimizer = optim.Adam(
            list(actor.parameters()), lr=self.args.algorithm.general.learning_rate
        )

        critic_optimizer = optim.Adam(
            list(critic.parameters()), lr=self.args.algorithm.general.learning_rate
        )

        return actor_optimizer, critic_optimizer

    def make_replay_buffer(self) -> ReplayBuffer:
        """
        Make replay buffer.

        Returns:
            ReplayBuffer: The replay buffer.
        """
        replay_buffer = ReplayBuffer(
            buffer_size=self.args.algorithm.general.buffer_size,
            observation_space=self.envs.single_observation_space,
            action_space=self.envs.single_action_space,
            device=self.device,
            handle_timeout_termination=False,
        )

        return replay_buffer

    @abstractmethod
    def create(self) -> LearningDDPG | LearningTD3:
        """
        Create learning algorithm.

        Raises:
            NotImplementedError: create method is not implemented.
        """
        raise NotImplementedError("create method is not implemented.")


class PreparationDDPG(PreparationInterface):
    def __init__(
        self,
        initial_sample: npt.NDArray[np.float64],
        initial_covariance: Optional[npt.NDArray[np.float64]] = None,
        initial_step_size: npt.NDArray[np.float64] = np.array([1.0]),
        log_mode: bool = True,
        log_target_pdf: Optional[
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        ] = None,
        grad_log_target_pdf: Optional[
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        ] = None,
        model_name: Optional[str] = None,
        posteriordb_path: Optional[str] = None,
        posterior_data: Optional[Dict[str, float | int | List[float | int]]] = None,
        hyperparameter_config_path: str = "",
        actor_config_path: str = "",
        critic_config_path: str = "",
        compile: bool = False,
    ) -> None:
        super().__init__(
            initial_sample,
            initial_covariance,
            initial_step_size,
            log_mode,
            log_target_pdf,
            grad_log_target_pdf,
            model_name,
            posteriordb_path,
            posterior_data,
            hyperparameter_config_path,
            actor_config_path,
            critic_config_path,
            compile,
        )
        self.qf1, self.target_qf1 = self.make_critic(compile)
        self.actor_optimizer, self.q_optimizer = self.make_optimizer(
            self.actor, self.qf1
        )

    def make_critic(self, compile: bool) -> Tuple[QNetwork, QNetwork]:
        """
        Make critic.

        Args:
            compile (bool): Whether to compile the model or not.

        Returns:
            Tuple[QNetwork, QNetwork]: The critic.
        """
        qf1 = QNetwork(self.envs, self.critic_config).to(self.device).double()
        target_qf1 = QNetwork(self.envs, self.critic_config).to(self.device).double()
        if compile:
            qf1 = torch.compile(qf1)
            target_qf1 = torch.compile(target_qf1)

        return qf1, target_qf1

    def create(self) -> LearningDDPG:
        """
        Create learning algorithm.

        Returns:
            LearningDDPG: DDPG algorithm.
        """
        return LearningDDPG(
            env=self.envs,
            predicted_env=self.predicted_envs,
            actor=self.actor,
            target_actor=self.target_actor,
            critic=self.qf1,
            target_critic=self.target_qf1,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.q_optimizer,
            replay_buffer=self.replay_buffer,
            learning_starts=self.args.algorithm.general.learning_starts,
            batch_size=self.args.algorithm.general.batch_size,
            exploration_noise=self.args.algorithm.specific.exploration_noise,
            gamma=self.args.algorithm.general.gamma,
            policy_frequency=self.args.algorithm.specific.policy_frequency,
            tau=self.args.algorithm.specific.tau,
            random_seed=self.args.experiments.seed,
            device=self.device,
            verbose=self.verbose,
        )


class PreparationTD3(PreparationInterface):
    def __init__(
        self,
        initial_sample: npt.NDArray[np.float64],
        initial_covariance: Optional[npt.NDArray[np.float64]] = None,
        initial_step_size: npt.NDArray[np.float64] = np.array([1.0]),
        log_mode: bool = True,
        log_target_pdf: Optional[
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        ] = None,
        grad_log_target_pdf: Optional[
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        ] = None,
        model_name: Optional[str] = None,
        posteriordb_path: Optional[str] = None,
        posterior_data: Optional[Dict[str, float | int | List[float | int]]] = None,
        hyperparameter_config_path: str = "",
        actor_config_path: str = "",
        critic_config_path: str = "",
        compile: bool = False,
    ) -> None:
        super().__init__(
            initial_sample,
            initial_covariance,
            initial_step_size,
            log_mode,
            log_target_pdf,
            grad_log_target_pdf,
            model_name,
            posteriordb_path,
            posterior_data,
            hyperparameter_config_path,
            actor_config_path,
            critic_config_path,
            compile,
        )
        self.qf1, self.target_qf1, self.qf2, self.target_qf2 = self.make_critic(compile)
        self.actor_optimizer, self.q_optimizer = self.make_optimizer(
            self.actor, self.qf1
        )

    def make_critic(
        self, compile: bool
    ) -> Tuple[QNetwork, QNetwork, QNetwork, QNetwork]:
        """
        Make critic.

        Args:
            compile (bool): Whether to compile the model or not.

        Returns:
            Tuple[QNetwork, QNetwork, QNetwork, QNetwork]: The critic.
        """
        qf1 = QNetwork(self.envs, self.critic_config).to(self.device).double()
        target_qf1 = QNetwork(self.envs, self.critic_config).to(self.device).double()
        qf2 = QNetwork(self.envs, self.critic_config).to(self.device).double()
        target_qf2 = QNetwork(self.envs, self.critic_config).to(self.device).double()
        if compile:
            qf1 = torch.compile(qf1)
            target_qf1 = torch.compile(target_qf1)
            qf2 = torch.compile(qf2)
            target_qf2 = torch.compile(target_qf2)

        return qf1, target_qf1, qf2, target_qf2

    def create(self) -> LearningTD3:
        return LearningTD3(
            env=self.envs,
            predicted_env=self.predicted_envs,
            actor=self.actor,
            target_actor=self.target_actor,
            critic=self.qf1,
            target_critic=self.target_qf1,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.q_optimizer,
            critic2=self.qf2,
            target_critic2=self.target_qf2,
            replay_buffer=self.replay_buffer,
            learning_starts=self.args.algorithm.general.learning_starts,
            batch_size=self.args.algorithm.general.batch_size,
            exploration_noise=self.args.algorithm.specific.exploration_noise,
            gamma=self.args.algorithm.general.gamma,
            policy_frequency=self.args.algorithm.specific.policy_frequency,
            tau=self.args.algorithm.specific.tau,
            random_seed=self.args.experiments.seed,
            device=self.device,
            verbose=self.verbose,
            policy_noise=self.args.algorithm.specific.policy_noise,
            noise_clip=self.args.algorithm.specific.noise_clip,
        )


class LearningAlgorithmFactory(ABC):
    @abstractmethod
    def create(self, **kwargs):
        """
        Create a learning algorithm instance.
        """
        raise NotImplementedError("create method is not implemented.")

    @staticmethod
    def register_algorithm(
        algorithm_name: str,
        hyperparameter_config_path: str,
        actor_config_path: str,
        critic_config_path: str,
    ):
        def decorator(cls):
            def create_instance(
                hyperparameter_config_path=hyperparameter_config_path,
                actor_config_path=actor_config_path,
                critic_config_path=critic_config_path,
                **kwargs,
            ):
                return cls(
                    hyperparameter_config_path=hyperparameter_config_path,
                    actor_config_path=actor_config_path,
                    critic_config_path=critic_config_path,
                ).create(**kwargs)

            LearningFactory.register_factory(algorithm_name, create_instance)
            return cls

        return decorator


class LearningFactory:
    _factories = {}

    @classmethod
    def register_factory(
        cls, algorithm: str, factory_callable: Callable[..., "LearningAlgorithmFactory"]
    ):
        cls._factories[algorithm.lower()] = factory_callable

    @classmethod
    def create_learning_instance(
        cls,
        algorithm: str,
        hyperparameter_config_path="",
        actor_config_path="",
        critic_config_path="",
        **kwargs,
    ):
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
    def __init__(
        self,
        hyperparameter_config_path: str,
        actor_config_path: str,
        critic_config_path: str,
    ):
        self.hyperparameter_config_path = hyperparameter_config_path
        self.actor_config_path = actor_config_path
        self.critic_config_path = critic_config_path

    def create(self, **kwargs):
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
    def __init__(
        self,
        hyperparameter_config_path: str,
        actor_config_path: str,
        critic_config_path: str,
    ):
        self.hyperparameter_config_path = hyperparameter_config_path
        self.actor_config_path = actor_config_path
        self.critic_config_path = critic_config_path

    def create(self, **kwargs):
        return PreparationDDPG(
            hyperparameter_config_path=self.hyperparameter_config_path,
            actor_config_path=self.actor_config_path,
            critic_config_path=self.critic_config_path,
            **kwargs,
        ).create()
