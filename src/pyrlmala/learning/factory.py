import json
import random
from abc import ABC, abstractmethod
from typing import Callable, Dict

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
from ..envs import BarkerEnv
from . import LearningDDPG


class LearningFactoryInterface(ABC):
    """Factory class for Learning.

    Args:
        log_target_pdf (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | str): Log target pdf.
        grad_log_target_pdf (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | str): Gradient of log target pdf.
        posterior_name (str): Posterior name.
        posteriordb_path (str): Posterior database path.
        compile (bool, optional): Compile model. Defaults to False.

    Returns:
        None
    """

    def __init__(
        self,
        log_target_pdf: (
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | str
        ),
        grad_log_target_pdf: (
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | str
        ),
        posteriordb_path: str | None = None,
        compile: bool = False,
        **kwargs,
    ) -> None:
        # Make log target pdf
        self.target_funs_factory(
            function=log_target_pdf,
            function_name="log_target_pdf",
            posteriordb_path=posteriordb_path,
            make_function=self.make_log_target_pdf,
            **kwargs,
        )

        # Make grad log target pdf
        self.target_funs_factory(
            function=grad_log_target_pdf,
            function_name="grad_log_target_pdf",
            posteriordb_path=posteriordb_path,
            make_function=self.make_grad_log_target_pdf,
            **kwargs,
        )

        # Make Args
        self.make_args(**kwargs)

        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.args.cuda else "cpu"
        )

        # Fixed random seed
        self.fixed_random_seed()

        # Make envs
        self.make_env()

        # Make Agent
        self.make_agent(compile=compile)

        # Make Replay Buffer
        self.make_replay_buffer()

    def fixed_random_seed(self):
        """
        Fixed random seed.
        """

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

    def target_funs_factory(
        self,
        function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | str,
        function_name: str,
        posteriordb_path: str,
        make_function: Callable[
            [str, str, Dict[str, float | int] | None],
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        ],
        **kwargs,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Factory function to create or return a target function (e.g., log target PDF or gradient log target PDF).

        Args:
            func (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | str): A callable representing the target function or
                a string representing the name of the posterior in PosteriorDB.
            func_name (str): The name of the function being processed, used for error messages.
            posteriordb_path (str): The path to the PosteriorDB database.
            make_function (Callable[[str, str, Dict[str, float | int] | None], Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]):
                A function used to create the target function from the posterior name.

        Raises:
            ValueError: If `func` is neither a callable nor a valid posterior name string.

        Returns:
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]: The callable target function.
        """
        if callable(function):
            return function
        elif isinstance(function, str):
            if posteriordb_path is None:
                raise ValueError(f"{function_name}: posteriordb_path must be provided.")
            else:
                return make_function(
                    posterior_name=function,
                    posteriordb_path=posteriordb_path,
                    posterior_data=kwargs.get("posterior_data"),
                )
        else:
            raise ValueError(f"{function_name} must be callable or str.")

    def make_log_target_pdf(
        self,
        posterior_name: str,
        posteriordb_path: str,
        posterior_data: Dict[str, float | int] | None = None,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Creates a log target probability density function (PDF) for the given posterior.

        Parameters:
        - posterior_name (str): Name of the posterior in the PosteriorDB.
        - posteriordb_path (str): Path to the PosteriorDB database.
        - posterior_data (Dict[str, float | int] | None, optional): Custom posterior data to be used.
            If None, the default data from PosteriorDB will be used.

        Returns:
        - Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]: A callable that computes the
            log density for given input.
        """

        # Load DataBase Locally
        pdb = PosteriorDatabase(posteriordb_path)

        # Load Dataset
        posterior = pdb.posterior(posterior_name)
        stan_code = posterior.model.stan_code_file_path()
        if posterior_data is None:
            stan_data = json.dumps(posterior.data.values())
        else:
            stan_data = json.dumps(posterior_data)

        # Return log_target_pdf
        model = bs.StanModel.from_stan_file(stan_code, stan_data)

        return model.log_density

    def make_grad_log_target_pdf(
        self,
        posterior_name: str,
        posteriordb_path: str,
        posterior_data: Dict[str, float | int] | None = None,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """
        Make the gradient of log target pdf.
        """

        # Load DataBase Locally
        pdb = PosteriorDatabase(posteriordb_path)

        # Load Dataset
        posterior = pdb.posterior(posterior_name)
        stan_code = posterior.model.stan_code_file_path()
        if posterior_data is None:
            stan_data = json.dumps(posterior.data.values())
        else:
            stan_data = json.dumps(posterior_data)

        # Return log_target_pdf
        model = bs.StanModel.from_stan_file(stan_code, stan_data)

        return lambda x: model.log_density_gradient(x)[1]

    def make_args(self, **kwargs) -> None:
        """
        Make Arguments.
        """
        # Initialize arguments
        args = Args()

        # Get all attributes
        args_attributes_dict = args.get_all_attributes()

        # Set attributes
        for key, value in kwargs.items():
            if key in args_attributes_dict.keys():
                if value is not None:
                    setattr(args, key, value)

        self.args = args

    def init_env(
        self,
        env_id: str | EnvSpec,
    ):
        def thunk():
            env = gym.make(
                id=env_id,
                log_target_pdf=self.log_target_pdf,
                sample_dim=self.args.sample_dim,
                total_timesteps=self.args.total_timesteps,
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(self.args.seed)

            return env

        return thunk

    def make_env(self):
        """
        Make environment and predict environment.
        """
        self.envs = gym.vector.SyncVectorEnv([self.init_env(env_id=self.args.env_id)])
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"
        self.envs.single_observation_space.dtype = np.float64

        self.predicted_envs = gym.vector.SyncVectorEnv(
            [self.init_env(env_id=self.args.env_id)]
        )
        assert isinstance(
            self.predicted_envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"
        self.predicted_envs.single_observation_space.dtype = np.float64

    def make_agent(self, compile: bool = False):
        self.actor = PolicyNetwork(self.envs).to(self.device).double()
        self.target_actor = PolicyNetwork(self.envs).to(self.device).double()
        self.qf1 = QNetwork(self.envs).to(self.device).double()
        self.target_qf1 = QNetwork(self.envs).to(self.device).double()

        if compile:
            self.actor = torch.compile(self.actor)
            self.target_actor = torch.compile(self.target_actor)
            self.qf1 = torch.compile(self.qf1)
            self.target_qf1 = torch.compile(self.qf1_target)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()), lr=self.args.learning_rate
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=self.args.learning_rate
        )

    def make_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.args.buffer_size,
            observation_space=self.envs.single_observation_space,
            action_space=self.envs.single_action_space,
            device=self.device,
            handle_timeout_termination=False,
        )

    @abstractmethod
    def create(self):
        raise NotImplementedError("create method is not implemented")


class LearningFactory(LearningFactoryInterface):
    def create(self, mode="ddpg"):
        match mode:
            case "ddpg":
                learning = self.make_DDPG()
            case "td3":
                learning = self.make_TD3()
            case _:
                raise NotImplementedError(
                    f"{mode} is not implemented, can only be ddpg or td3."
                )

        return learning

    def make_DDPG(self):
        learning = LearningDDPG(
            env=self.envs,
            actor=self.actor,
            target_actor=self.target_actor,
            critic=self.qf1,
            target_critic=self.target_qf1,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.q_optimizer,
            replay_buffer=self.replay_buffer,
            total_timesteps=self.args.total_timesteps,
            learning_starts=self.args.learning_starts,
            batch_size=self.args.batch_size,
            exploration_noise=self.args.exploration_noise,
            gamma=self.args.gamma,
            policy_frequency=self.args.policy_frequency,
            tau=self.args.tau,
            seed=self.args.seed,
            device=self.device,
        )
        return learning

    def make_TD3(self):
        pass
