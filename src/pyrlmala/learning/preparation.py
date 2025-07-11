import json
import random
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import bridgestan as bs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium.envs.registration import EnvSpec
from posteriordb import PosteriorDatabase
from stable_baselines3.common.buffers import ReplayBuffer
from torch import optim

from ..agent import PolicyNetwork, QNetwork
from ..config import (
    HyperparameterConfigParser,
    PolicyNetworkConfigParser,
    QNetworkConfigParser,
)
from ..envs import MCMCEnvBase
from .learning import LearningDDPG, LearningTD3
from .pretrain import PretrainFactory, PretrainMockDataset, PretrainPosteriorDBDataset

T = TypeVar("T")


class PosteriorDBFunctionsGenerator:
    """
    Target functions generator for PosteriorDB.

    Attributes:
        model_name (str): The name of the model in PosteriorDB.
        posteriordb_path (str): The path to the PosteriorDB database.
        posterior_data (Optional[Dict[str, float | int | List[float | int]]]): The parameter of the model.
        stan_model (bs.StanModel): The Stan model.
        log_pdf (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): The log probability density function.
        grad_log_pdf (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): The gradient of the log probability density function.
    """

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
    """
    Factory class for creating learning algorithms based on reinforcement learning strategies.

    Attributes:
        initial_sample (npt.NDArray[np.float64]): The initial sample.
        initial_covariance (Optional[npt.NDArray[np.float64]]): The initial covariance. Defaults to None.
        initial_step_size (npt.NDArray[np.float64]): The initial step size. Defaults to np.array([1.0]).
        log_mode (bool): The log mode. Defaults to True.
        log_target_pdf (Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]): The log probability density function of the target distribution. Defaults to None.
        grad_log_target_pdf (Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]): The gradient of the log probability density function. Defaults to None.
        model_name (Optional[str]): The model name in PosteriorDB. Defaults to None.
        posteriordb_path (Optional[str]): The path of PosteriorDB. Defaults to None.
        posterior_data (Optional[Dict[str, float  |  int  |  List[float  |  int]]]): The posterior data. Defaults to None.
        hyperparameter_config_path (str): The path to the hyperparameter configuration file.
        actor_config_path (str): The path to the actor configuration file.
        critic_config_path (str): The path to the critic configuration file.
        compile (bool): Whether to compile the model or not. Defaults to False.
        verbose (bool): Whether to show the verbose message or not. Defaults to True.
        args (HyperparameterConfigParser): The hyperparameter configuration parser.
        actor_config (PolicyNetworkConfigParser): The actor configuration parser.
        critic_config (QNetworkConfigParser): The critic configuration parser.
        device (torch.device): The device.
        envs (gym.vector.SyncVectorEnv): The environment.
        predicted_envs (gym.vector.SyncVectorEnv): The predict environment.
        actor (PolicyNetwork): The actor.
        replay_buffer (ReplayBuffer): The replay buffer.
    """

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
            hyperparameter_config_path (str, optional): The path to the hyperparameter configuration file. Defaults to "".
            actor_config_path (str, optional): The path to the actor configuration file. Defaults to "".
            critic_config_path (str, optional): The path to the critic configuration file. Defaults to "".
            compile (bool, optional): Whether to compile the model or not. Defaults to False.
            verbose (bool, optional): Whether to show the verbose message or not. Defaults to True.

        Raises:
            ValueError: If log_target_pdf or grad_log_target_pdf is not provided, model_name and posteriordb_path cannot be None.
        """
        # Initial Essential Attribution of Environment
        self.initial_sample = initial_sample
        if initial_covariance is None:
            self.initial_covariance = np.eye(len(initial_sample.flatten()))
        else:
            self.initial_covariance = initial_covariance
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
        self.actor, self.target_actor = self.make_actor(
            compile,
            model_name,
            posteriordb_path,
        )

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
            hyperparameter_config_path (str): The path to the hyperparameter configuration file.
            actor_config_path (str): The path to the actor configuration file.
            critic_config_path (str): The path to the critic configuration file.

        Raises:
            ValueError: hyperparameter_config_path must be provided.
            ValueError: actor_config_path must be provided.
            ValueError: critic_config_path must be provided.

        Returns:
            Tuple[HyperparameterConfigParser, PolicyNetworkConfigParser, QNetworkConfigParser]: The arguments.
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
        self, env_id: str | EnvSpec, total_timesteps: int, max_steps_per_episode: int
    ) -> Callable[[], MCMCEnvBase]:
        """
        Initialize environment to the function.

        Args:
            env_id (str | EnvSpec): The environment id.

        Returns:
            Callable[[], MCMCEnvBase]: The thunk function.
        """

        def thunk() -> MCMCEnvBase:
            """
            Initialize environment.

            Returns:
                MCMCEnvBase: The environment.
            """
            env = gym.make(
                id=env_id,
                log_target_pdf_unsafe=self.log_target_pdf,
                grad_log_target_pdf_unsafe=self.grad_log_target_pdf,
                initial_sample=self.initial_sample,
                initial_covariance=self.initial_covariance,
                initial_step_size=self.initial_step_size,
                total_timesteps=total_timesteps,
                max_steps_per_episode=max_steps_per_episode,
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
                    max_steps_per_episode=self.args.algorithm.general.max_steps_per_episode,
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
                    max_steps_per_episode=self.args.algorithm.general.max_steps_per_episode,
                )
            ]
        )
        assert isinstance(
            predicted_envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"
        predicted_envs.single_observation_space.dtype = np.float64

        return envs, predicted_envs

    def get_actor_input_size(self) -> int:
        return self.envs.single_observation_space.shape[0] >> 1

    def make_actor(
        self,
        compile: bool,
        model_name: Optional[str],
        posteriordb_path: Optional[str],
    ) -> Tuple[PolicyNetwork, PolicyNetwork]:
        """
        Make actor.

        Args:
            compile (bool): Whether to compile the model or not.

        Returns:
            Tuple[PolicyNetwork, PolicyNetwork]: The actor.
        """
        input_size = self.get_actor_input_size()
        actor = PolicyNetwork(input_size, self.actor_config).to(self.device).double()
        target_actor = (
            PolicyNetwork(input_size, self.actor_config).to(self.device).double()
        )

        if compile:
            actor = torch.compile(actor)
            target_actor = torch.compile(target_actor)

        if self.args.algorithm.general.actor_pretrain:
            initial_sample_torch = (
                torch.from_numpy(self.initial_sample).to(self.device).double()
            )

            if model_name is None or posteriordb_path is None:
                mock_dataset = PretrainMockDataset(
                    initial_sample=initial_sample_torch,
                    step_size=self.initial_step_size.item(),
                    num_data=self.args.algorithm.general.actor_pretrain_num_data,
                    mag=self.args.algorithm.general.actor_pretrain_mag,
                )
            else:
                mock_dataset = PretrainPosteriorDBDataset(
                    model_name=model_name,
                    posteriordb_path=posteriordb_path,
                    step_size=self.initial_step_size.item(),
                )
            actor = PretrainFactory.train(
                actor,
                mock_dataset,
                num_epochs=self.args.algorithm.general.actor_pretrain_num_epochs,
                batch_size=self.args.algorithm.general.actor_pretrain_batch_size,
                shuffle=self.args.algorithm.general.actor_pretrain_shuffle,
                device=self.device,
                verbose=self.verbose,
            )

        # Align target actor with the actor
        target_actor.load_state_dict(actor.state_dict())

        return actor, target_actor

    def get_critic_input_size(self) -> int:
        return (
            self.envs.single_observation_space.shape[0]
            + self.envs.single_action_space.shape[0]
        )

    @abstractmethod
    def make_critic(
        self, compile: bool
    ) -> Tuple[QNetwork, QNetwork] | Tuple[QNetwork, QNetwork, QNetwork, QNetwork]:
        """
        Make critic.

        Args:
            compile (bool): Whether to compile the model or not.

        Raises:
            NotImplementedError: make_critic method is not implemented.

        Returns:
            Tuple[QNetwork, QNetwork] | Tuple[QNetwork, QNetwork, QNetwork, QNetwork]: The critic.
        """
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
            list(actor.parameters()), lr=self.args.algorithm.general.actor_learning_rate
        )

        critic_optimizer = optim.Adam(
            list(critic.parameters()),
            lr=self.args.algorithm.general.critic_learning_rate,
        )

        return actor_optimizer, critic_optimizer

    def make_scheduler(
        self,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
    ) -> Tuple[
        optim.lr_scheduler.LRScheduler | None, optim.lr_scheduler.LRScheduler | None
    ]:
        """
        Make scheduler. If the scheduler is not provided, it will return None.

        Args:
            actor_optimizer (optim.Optimizer): The actor optimizer.
            critic_optimizer (optim.Optimizer): The critic optimizer.

        Returns:
            Tuple[ optim.lr_scheduler.LRScheduler | None, optim.lr_scheduler.LRScheduler | None ]: The scheduler. If the scheduler is not provided, it will return None.
        """
        actor_scheduler, critic_scheduler = None, None

        if actor_scheduler_type := getattr(self.actor_config.scheduler, "type"):
            actor_scheduler = getattr(optim.lr_scheduler, actor_scheduler_type)(
                actor_optimizer, **getattr(self.actor_config.scheduler, "params")
            )

        if critic_scheduler_type := getattr(self.critic_config.scheduler, "type"):
            critic_scheduler = getattr(optim.lr_scheduler, critic_scheduler_type)(
                critic_optimizer, **getattr(self.critic_config.scheduler, "params")
            )

        return actor_scheduler, critic_scheduler

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
    """
    Factory class for creating learning algorithms based on reinforcement learning strategies.

    Attributes:
        initial_sample (npt.NDArray[np.float64]): The initial sample.
        initial_covariance (Optional[npt.NDArray[np.float64]]): The initial covariance. Defaults to None.
        initial_step_size (npt.NDArray[np.float64]): The initial step size. Defaults to np.array([1.0]).
        log_mode (bool): The log mode. Defaults to True.
        log_target_pdf (Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]): The log probability density function of the target distribution. Defaults to None.
        grad_log_target_pdf (Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]): The gradient of the log probability density function. Defaults to None.
        model_name (Optional[str]): The model name in PosteriorDB. Defaults to None.
        posteriordb_path (Optional[str]): The path of PosteriorDB. Defaults to None.
        posterior_data (Optional[Dict[str, float  |  int  |  List[float  |  int]]]): The posterior data. Defaults to None.
        hyperparameter_config_path (str): The path to the hyperparameter configuration file.
        actor_config_path (str): The path to the actor configuration file.
        critic_config_path (str): The path to the critic configuration file.
        compile (bool): Whether to compile the model or not. Defaults to False.
        verbose (bool): Whether to show the verbose message or not. Defaults to True.
        args (HyperparameterConfigParser): The hyperparameter configuration parser.
        actor_config (PolicyNetworkConfigParser): The actor configuration parser.
        critic_config (QNetworkConfigParser): The critic configuration parser.
        device (torch.device): The device.
        envs (gym.vector.SyncVectorEnv): The environment.
        predicted_envs (gym.vector.SyncVectorEnv): The predict environment.
        actor (PolicyNetwork): The actor.
        replay_buffer (ReplayBuffer): The replay buffer.
        qf1 (QNetwork): The critic.
        target_qf1 (QNetwork): The target critic.
        actor_optimizer (optim.Optimizer): The actor optimizer.
        q_optimizer (optim.Optimizer): The critic optimizer.
    """

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
            initial_sample (npt.NDArray[np.float64]): The initial sample.
            initial_covariance (Optional[npt.NDArray[np.float64]], optional): The initial covariance. Defaults to None.
            initial_step_size (npt.NDArray[np.float64], optional): The initial step size. Defaults to np.array([1.0]).
            log_mode (bool, optional): The log mode. Defaults to True.
            log_target_pdf (Optional[ Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] ], optional): The log probability density function of the target distribution. Defaults to None.
            grad_log_target_pdf (Optional[ Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] ], optional): The gradient of the log probability density function. Defaults to None.
            model_name (Optional[str], optional):The model name in PosteriorDB. Defaults to None.
            posteriordb_path (Optional[str], optional): The path of PosteriorDB. Defaults to None.
            posterior_data (Optional[Dict[str, float  |  int  |  List[float  |  int]]], optional): The posterior data. Defaults to None.
            hyperparameter_config_path (str, optional): The path to the hyperparameter configuration file. Defaults to "".
            actor_config_path (str, optional):The path to the actor configuration file. Defaults to "".
            critic_config_path (str, optional): The path to the critic configuration file. Defaults to "".
            compile (bool, optional): Whether to compile the model or not. Defaults to False.
            verbose (bool, optional): Whether to show the verbose message or not. Defaults to True.
        """
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
            verbose,
        )
        self.qf1, self.target_qf1 = self.make_critic(compile)
        self.actor_optimizer, self.q_optimizer = self.make_optimizer(
            self.actor, self.qf1
        )
        self.actor_scheduler, self.q_scheduler = self.make_scheduler(
            self.actor_optimizer, self.q_optimizer
        )

    def make_critic(self, compile: bool) -> Tuple[QNetwork, QNetwork]:
        """
        Make critic.

        Args:
            compile (bool): Whether to compile the model or not.

        Returns:
            Tuple[QNetwork, QNetwork]: The critic.
        """
        input_size = self.get_critic_input_size()
        qf1 = QNetwork(input_size, self.critic_config).to(self.device).double()
        target_qf1 = QNetwork(input_size, self.critic_config).to(self.device).double()
        if compile:
            qf1 = torch.compile(qf1)
            target_qf1 = torch.compile(target_qf1)

        # Align target critic with the critic
        target_qf1.load_state_dict(qf1.state_dict())

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
            actor_scheduler=self.actor_scheduler,
            actor_gradient_clipping=self.args.algorithm.general.actor_gradient_clipping,
            actor_gradient_threshold=self.args.algorithm.general.actor_gradient_threshold,
            actor_gradient_norm=self.args.algorithm.general.actor_gradient_norm,
            critic_scheduler=self.q_scheduler,
            critic_gradient_clipping=self.args.algorithm.general.critic_gradient_clipping,
            critic_gradient_threshold=self.args.algorithm.general.critic_gradient_threshold,
            critic_gradient_norm=self.args.algorithm.general.critic_gradient_norm,
            learning_starts=self.args.algorithm.general.learning_starts,
            batch_size=self.args.algorithm.general.batch_size,
            exploration_noise=self.args.algorithm.specific.exploration_noise,
            gamma=self.args.algorithm.general.gamma,
            policy_frequency=self.args.algorithm.specific.policy_frequency,
            tau=self.args.algorithm.specific.tau,
            random_seed=self.args.experiments.seed,
            num_of_top_policies=self.args.experiments.num_of_top_policies,
            r_bar=self.args.algorithm.general.r_bar,
            r_bar_alpha=self.args.algorithm.general.r_bar_alpha,
            device=self.device,
            track=self.args.experiments.track,
            verbose=self.verbose,
            run_name=f"{self.args.algorithm.general.env_id}__{self.args.experiments.exp_name}__{self.args.experiments.seed}__{int(time.time())}",
        )


class PreparationTD3(PreparationInterface):
    """
    Factory class for creating learning algorithms based on reinforcement learning strategies.

    Attributes:
        initial_sample (npt.NDArray[np.float64]): The initial sample.
        initial_covariance (Optional[npt.NDArray[np.float64]]): The initial covariance. Defaults to None.
        initial_step_size (npt.NDArray[np.float64]): The initial step size. Defaults to np.array([1.0]).
        log_mode (bool): The log mode. Defaults to True.
        log_target_pdf (Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]): The log probability density function of the target distribution. Defaults to None.
        grad_log_target_pdf (Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]): The gradient of the log probability density function. Defaults to None.
        model_name (Optional[str]): The model name in PosteriorDB. Defaults to None.
        posteriordb_path (Optional[str]): The path of PosteriorDB. Defaults to None.
        posterior_data (Optional[Dict[str, float  |  int  |  List[float  |  int]]]): The posterior data. Defaults to None.
        hyperparameter_config_path (str): The path to the hyperparameter configuration file.
        actor_config_path (str): The path to the actor configuration file.
        critic_config_path (str): The path to the critic configuration file.
        compile (bool): Whether to compile the model or not. Defaults to False.
        verbose (bool): Whether to show the verbose message or not. Defaults to True.
        args (HyperparameterConfigParser): The hyperparameter configuration parser.
        actor_config (PolicyNetworkConfigParser): The actor configuration parser.
        critic_config (QNetworkConfigParser): The critic configuration parser.
        device (torch.device): The device.
        envs (gym.vector.SyncVectorEnv): The environment.
        predicted_envs (gym.vector.SyncVectorEnv): The predict environment.
        actor (PolicyNetwork): The actor.
        replay_buffer (ReplayBuffer): The replay buffer.
        qf1 (QNetwork): The critic.
        target_qf1 (QNetwork): The target critic.
        qf2 (QNetwork): The critic.
        target_qf2 (QNetwork): The target critic.
        actor_optimizer (optim.Optimizer): The actor optimizer.
        q_optimizer (optim.Optimizer): The critic optimizer.
    """

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
            initial_sample (npt.NDArray[np.float64]): The initial sample.
            initial_covariance (Optional[npt.NDArray[np.float64]], optional): The initial covariance. Defaults to None.
            initial_step_size (npt.NDArray[np.float64], optional): The initial step size. Defaults to np.array([1.0]).
            log_mode (bool, optional): The log mode. Defaults to True.
            log_target_pdf (Optional[ Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] ], optional): The log probability density function of the target distribution. Defaults to None.
            grad_log_target_pdf (Optional[ Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] ], optional): The gradient of the log probability density function. Defaults to None.
            model_name (Optional[str], optional):The model name in PosteriorDB. Defaults to None.
            posteriordb_path (Optional[str], optional): The path of PosteriorDB. Defaults to None.
            posterior_data (Optional[Dict[str, float  |  int  |  List[float  |  int]]], optional): The posterior data. Defaults to None.
            hyperparameter_config_path (str, optional): The path to the hyperparameter configuration file. Defaults to "".
            actor_config_path (str, optional):The path to the actor configuration file. Defaults to "".
            critic_config_path (str, optional): The path to the critic configuration file. Defaults to "".
            compile (bool, optional): Whether to compile the model or not. Defaults to False.
            verbose (bool, optional): Whether to show the verbose message or not. Defaults to True.
        """
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
            verbose,
        )
        self.qf1, self.target_qf1, self.qf2, self.target_qf2 = self.make_critic(compile)
        self.actor_optimizer, self.q_optimizer = self.make_optimizer(
            self.actor, self.qf1
        )
        self.actor_scheduler, self.q_scheduler = self.make_scheduler(
            self.actor_optimizer, self.q_optimizer
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
        input_size = self.get_critic_input_size()
        qf1 = QNetwork(input_size, self.critic_config).to(self.device).double()
        target_qf1 = QNetwork(input_size, self.critic_config).to(self.device).double()
        qf2 = QNetwork(input_size, self.critic_config).to(self.device).double()
        target_qf2 = QNetwork(input_size, self.critic_config).to(self.device).double()
        if compile:
            qf1 = torch.compile(qf1)
            target_qf1 = torch.compile(target_qf1)
            qf2 = torch.compile(qf2)
            target_qf2 = torch.compile(target_qf2)

        # Align target critic with the critic
        target_qf1.load_state_dict(qf1.state_dict())
        target_qf2.load_state_dict(qf2.state_dict())

        return qf1, target_qf1, qf2, target_qf2

    def create(self) -> LearningTD3:
        """
        Create learning algorithm.

        Returns:
            LearningTD3: TD3 algorithm.
        """
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
            actor_scheduler=self.actor_scheduler,
            actor_gradient_clipping=self.args.algorithm.general.actor_gradient_clipping,
            actor_gradient_threshold=self.args.algorithm.general.actor_gradient_threshold,
            actor_gradient_norm=self.args.algorithm.general.actor_gradient_norm,
            critic_scheduler=self.q_scheduler,
            critic_gradient_clipping=self.args.algorithm.general.critic_gradient_clipping,
            critic_gradient_threshold=self.args.algorithm.general.critic_gradient_threshold,
            critic_gradient_norm=self.args.algorithm.general.critic_gradient_norm,
            learning_starts=self.args.algorithm.general.learning_starts,
            batch_size=self.args.algorithm.general.batch_size,
            exploration_noise=self.args.algorithm.specific.exploration_noise,
            gamma=self.args.algorithm.general.gamma,
            policy_frequency=self.args.algorithm.specific.policy_frequency,
            tau=self.args.algorithm.specific.tau,
            random_seed=self.args.experiments.seed,
            num_of_top_policies=self.args.experiments.num_of_top_policies,
            r_bar=self.args.algorithm.general.r_bar,
            r_bar_alpha=self.args.algorithm.general.r_bar_alpha,
            device=self.device,
            verbose=self.verbose,
            track=self.args.experiments.track,
            policy_noise=self.args.algorithm.specific.policy_noise,
            noise_clip=self.args.algorithm.specific.noise_clip,
            run_name=f"{self.args.algorithm.general.env_id}__{self.args.experiments.exp_name}__{self.args.experiments.seed}__{int(time.time())}",
        )
