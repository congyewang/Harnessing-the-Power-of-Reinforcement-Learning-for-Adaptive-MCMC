import json
import os
import re
from typing import Callable, Dict, Union

import bridgestan as bs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.envs.registration import EnvSpec
from numpy.typing import NDArray
from posteriordb import PosteriorDatabase
from scipy.stats._multivariate import _PSD


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
