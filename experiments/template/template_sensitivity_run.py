import gc
from typing import Tuple

import numpy as np
from loguru import logger
from numpy import typing as npt

from pyrlmala.learning import LearningFactory
from pyrlmala.utils import Toolbox
from pyrlmala.utils.posteriordb import PosteriorDBToolbox
from pyrlmala.utils.target import AutoStanTargetPDF

model_name = "{{ model_name }}"
posteriordb_path = "{{ posteriordb_path }}"
repeat_num = {{repeat_num}}


def output_initial(
    model_name: str, l: float = 1.65
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Output a smooth initial sample, covariance matrix, and step size based on model dimension using log-linear interpolation.

    Args:
        model_name (str): The name of the model.
        l (float, optional): The scaling factor. Defaults to 1.65.

    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            A tuple containing the initial sample, covariance matrix, and step size.
    """
    gs = Toolbox.gold_standard(model_name, posteriordb_path)
    target = AutoStanTargetPDF(model_name, posteriordb_path)
    sigma_inv = -target.hess_log_target_pdf(gs.mean(axis=0))
    d = sigma_inv.shape[0]

    lam_max = np.linalg.eigvalsh(sigma_inv).max()
    eps0 = l / np.sqrt(lam_max * d ** (1 / 3))
    initial_step_size = 29.0168 * eps0**3 - 25.6180 * eps0**2 + 3.0239 * eps0 + 1.3476

    hessian_matrix = target.hess_log_target_pdf(gs.mean(axis=0))
    initial_covariance_matrix = -np.linalg.inv(hessian_matrix)

    return gs[0], initial_covariance_matrix, initial_step_size.reshape(1)


def get_samples(
    random_seed: int,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    pdb_toolbox = PosteriorDBToolbox(posteriordb_path)
    gs = pdb_toolbox.get_gold_standard(model_name)

    sample_dim = gs.shape[1]
    initial_sample, initial_covariance, initial_step_size = output_initial(model_name)
    algorithm = "{{ rl_algorithm }}"
    mcmc_env = "{{ mcmc_env }}"

    learning_instance = LearningFactory.create_learning_instance(
        algorithm=algorithm,
        model_name=model_name,
        posteriordb_path=posteriordb_path,
        initial_sample=initial_sample,
        initial_covariance=initial_covariance,
        initial_step_size=initial_step_size,
        hyperparameter_config_path=f"./config/{algorithm}_{mcmc_env}/{algorithm}_{mcmc_env}_seed_{random_seed}.toml",
        actor_config_path="./config/actor.toml",
        critic_config_path="./config/critic.toml",
    )

    learning_instance.train()
    learning_instance.predict()

    predicted_sample = learning_instance.predicted_observation[:, 0:sample_dim]

    del learning_instance
    gc.collect()

    return gs, predicted_sample


def write_header(file_path: str) -> None:
    with open(file_path, "w") as file:
        file.write("model_name,rl_algorithm,mcmc_env,random_seed,mmd\n")


def write_results(random_seed: int, mmd: float, file_path: str) -> None:
    rl_algorithm = "{{ rl_algorithm }}"
    mcmc_env = "{{ mcmc_env }}"

    with open(file_path, "a+") as file:
        file.write(f"{model_name},{rl_algorithm},{mcmc_env},{random_seed},{mmd}\n")


def main():
    file_path = "{{ model_name }}_{{ rl_algorithm }}_{{ mcmc_env }}.csv"
    write_header(file_path)

    for random_number in range(repeat_num):
        try:
            gs, predicted_sample = get_samples(random_number)
            median_lengthscale = Toolbox.median_trick(gs)
            mmd = Toolbox.calculate_mmd(
                gs, predicted_sample[-5_000:], median_lengthscale
            )
            write_results(random_number, mmd, file_path)
        except Exception as e:
            logger.error(f"Error occurred for random seed {random_number}: {e}")
            continue


if __name__ == "__main__":
    main()
