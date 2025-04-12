import gc
from typing import Tuple

import numpy as np
from numpy import typing as npt

from pyrlmala.learning import LearningFactory
from pyrlmala.utils import Toolbox
from pyrlmala.utils.posteriordb import PosteriorDBToolbox

model_name = "{{ model_name }}"
posteriordb_path = "{{ posteriordb_path }}"


def output_initial_step_size(model_sample_dim: int) -> float:
    """
    Output a smooth initial step size based on model dimension using log-linear interpolation.
    """
    if model_sample_dim <= 10:
        return 0.5
    elif model_sample_dim >= 100:
        return 0.01
    else:
        # Log-linear interpolate between 0.5 (dim=10) and 0.01 (dim=100)
        log_eps = np.interp(
            np.log(model_sample_dim),
            [np.log(10), np.log(100)],
            [np.log(0.5), np.log(0.01)],
        )
        return float(np.exp(log_eps))


def get_samples() -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    pdb_toolbox = PosteriorDBToolbox(posteriordb_path)
    gs = pdb_toolbox.get_gold_standard(model_name)

    sample_dim = gs.shape[1]
    initial_sample = gs[0]
    initial_step_size = np.array([output_initial_step_size(sample_dim)])
    initial_covariance = pdb_toolbox.get_fisher_information_matrix(model_name)
    algorithm = "{{ rl_algorithm }}"

    learning_instance = LearningFactory.create_learning_instance(
        algorithm=algorithm,
        model_name=model_name,
        posteriordb_path=posteriordb_path,
        initial_sample=initial_sample,
        initial_covariance=initial_covariance,
        initial_step_size=initial_step_size,
        hyperparameter_config_path="{{ hyperparameter_config_path }}",
        actor_config_path="{{ actor_config_path }}",
        critic_config_path="{{ critic_config_path }}",
    )

    learning_instance.train()
    learning_instance.predict()

    predicted_sample = learning_instance.predicted_observation[:, 0:sample_dim]

    del learning_instance
    gc.collect()

    return gs, predicted_sample


def write_results(mmd: float) -> None:
    rl_algorithm = "{{ rl_algorithm }}"
    mcmc_env = "{{ mcmc_env }}"
    random_seed = {{random_seed}}

    with open(
        "{{ model_name }}_{{ rl_algorithm }}_{{ mcmc_env }}_seed_{{ random_seed }}.csv",
        "w",
    ) as file:
        file.write("model_name,rl_algorithm,mcmc_env,random_seed,mmd\n")
        file.write(f"{model_name},{rl_algorithm},{mcmc_env},{random_seed},{mmd}\n")


def main():
    gs, predicted_sample = get_samples()

    # Calculate the MMD
    mmd = Toolbox.calculate_mmd(gs, predicted_sample)
    write_results(mmd)


if __name__ == "__main__":
    main()
