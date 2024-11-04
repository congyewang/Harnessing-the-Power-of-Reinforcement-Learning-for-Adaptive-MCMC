import json
import os
import re
from dataclasses import dataclass, fields
from typing import Callable, Dict, Union

import bridgestan as bs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.envs.registration import EnvSpec
from numpy.typing import NDArray
from posteriordb import PosteriorDatabase
from scipy.stats._multivariate import _PSD


@dataclass
class Args:
    exp_name: str = "RLBarker"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RLBarker"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BarkerEnv-v1.0"
    """the environment id of the Atari game"""
    total_timesteps: int = int(1e3)
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 24
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 4
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    def get_all_attributes(self):
        """
        get all attributes of this class
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}


class Toolbox:
    @staticmethod
    def make_env(
        env_id: Union[str, EnvSpec],
        log_target_pdf: Callable[
            [Union[float, np.float64, npt.NDArray[np.float64]]],
            Union[float, np.float64],
        ],
        grad_log_target_pdf: Callable[
            [Union[float, np.float64, npt.NDArray[np.float64]]],
            Union[float, np.float64],
        ],
        initial_sample: Union[np.float64, npt.NDArray[np.float64]],
        initial_covariance: Union[np.float64, npt.NDArray[np.float64], None] = None,
        total_timesteps: int = 500_000,
        log_mode: bool = True,
        seed: int = 42,
    ):
        def thunk():
            env = gym.make(
                env_id,
                log_target_pdf_unsafe=log_target_pdf,
                grad_log_target_pdf_unsafe=grad_log_target_pdf,
                initial_sample=initial_sample,
                initial_covariance=initial_covariance,
                total_timesteps=total_timesteps,
                log_mode=log_mode,
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)

            return env

        return thunk

    @classmethod
    def nearestPD(cls, A: NDArray[np.float64]):
        """
        Find the nearest positive-definite matrix to input
        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if cls.isPD(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not cls.isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3

    @classmethod
    def isPD(cls, B: NDArray[np.float64]):
        """
        Returns true when input is positive-definite, via Cholesky, det, and _PSD from scipy.
        """
        try:
            _ = np.linalg.cholesky(B)
            res_cholesky = True
        except np.linalg.LinAlgError:
            res_cholesky = False

        try:
            assert np.linalg.det(B) > 0, "Determinant is negative"
            res_det = True
        except AssertionError:
            res_det = False

        try:
            _PSD(B, allow_singular=False)
            res_PSD = True
        except Exception as e:
            if re.search("[Pp]ositive", str(e)):
                return False
            else:
                raise

        res = res_cholesky and res_det and res_PSD

        return res

    @staticmethod
    def make_log_target_pdf(
        posterior_name: str,
        posteriordb_path: str,
        posterior_data: Union[Dict[str, Union[float, int]], None] = None,
    ):
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

    @staticmethod
    def create_folder(file_path: str) -> None:
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
