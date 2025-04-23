import gc
from typing import Tuple

import numpy as np
from numpy import typing as npt

from pyrlmala.learning import LearningFactory
from pyrlmala.utils import Toolbox
from pyrlmala.utils.posteriordb import PosteriorDBToolbox

model_name = "{{ model_name }}"
posteriordb_path = "{{ posteriordb_path }}"
repeat_num = {{repeat_num}}


def output_initial_step_size(model_name: str) -> float:
    """
    Output a smooth initial step size based on model dimension using log-linear interpolation.

    Args:
        model_name (str): The name of the model.

    Returns:
        float: The initial step size for the model.
    """
    if model_name == "kidiq_with_mom_work-kidscore_mom_work":
        step_size = 2.96e-02
    elif model_name == "bball_drive_event_0-hmm_drive_0":
        step_size = 3.36e-02
    elif model_name == "kilpisjarvi_mod-kilpisjarvi":
        step_size = 3.75e-02
    elif model_name == "kidiq-kidscore_momhsiq":
        step_size = 6.66e-03
    elif model_name == "eight_schools-eight_schools_noncentered":
        step_size = 5.64e-01
    elif model_name == "nes2000-nes":
        step_size = 8.59e-04
    elif model_name == "nes1996-nes":
        step_size = 2.91e-06
    elif model_name == "earnings-logearn_interaction":
        step_size = 3.06e-02
    elif model_name == "arK-arK":
        step_size = 4.37e-05
    elif model_name == "gp_pois_regr-gp_regr":
        step_size = 8.78e-02
    elif model_name == "diamonds-diamonds":
        step_size = 1.02e-07
    elif model_name == "garch-garch11":
        step_size = 2.00e-03
    elif model_name == "mesquite-logmesquite":
        step_size = 3.29e-04
    elif model_name == "mesquite-logmesquite_logva":
        step_size = 2.15e-02
    elif model_name == "nes1992-nes":
        step_size = 1.08e-07
    elif model_name == "nes1976-nes":
        step_size = 9.61e-08
    elif model_name == "sblri-blr":
        step_size = 1.41e-02
    elif model_name == "earnings-log10earn_height":
        step_size = 7.81e-02
    elif model_name == "earnings-logearn_height_male":
        step_size = 7.74e-03
    elif model_name == "nes1988-nes":
        step_size = 1.04e-07
    elif model_name == "mesquite-logmesquite_logvas":
        step_size = 2.56e-04
    elif model_name == "earnings-earn_height":
        step_size = 1.70e-05
    elif model_name == "mcycle_gp-accel_gp":
        step_size = 5.64e-01
    elif model_name == "hmm_example-hmm_example":
        step_size = 4.26e-03
    elif model_name == "bball_drive_event_1-hmm_drive_1":
        step_size = 1.38e-02
    elif model_name == "nes1980-nes":
        step_size = 8.29e-04
    elif model_name == "earnings-logearn_height":
        step_size = 7.24e-02
    elif model_name == "one_comp_mm_elim_abs-one_comp_mm_elim_abs":
        step_size = 1.37e-06
    elif model_name == "kidiq-kidscore_momhs":
        step_size = 8.16e-02
    elif model_name == "earnings-logearn_logheight_male":
        step_size = 2.67e-02
    elif model_name == "kidiq-kidscore_momiq":
        step_size = 1.71e-01
    elif model_name == "hudson_lynx_hare-lotka_volterra":
        step_size = 1.93e-04
    elif model_name == "nes1972-nes":
        step_size = 1.02e-07
    elif model_name == "sblrc-blr":
        step_size = 2.68e-05
    elif model_name == "kidiq_with_mom_work-kidscore_interaction_c":
        step_size = 3.08e-02
    elif model_name == "nes1984-nes":
        step_size = 8.99e-08
    elif model_name == "eight_schools-eight_schools_centered":
        step_size = 6.33e-08
    elif model_name == "earnings-logearn_interaction_z":
        step_size = 3.18e-02
    elif model_name == "mesquite-logmesquite_logvash":
        step_size = 1.91e-05
    elif model_name == "mesquite-logmesquite_logvolume":
        step_size = 4.36e-02
    elif model_name == "kidiq-kidscore_interaction":
        step_size = 3.07e-02
    elif model_name == "arma-arma11":
        step_size = 5.06e-03
    elif model_name == "kidiq_with_mom_work-kidscore_interaction_c2":
        step_size = 1.24e-03
    elif model_name == "kidiq_with_mom_work-kidscore_interaction_z":
        step_size = 3.05e-02
    elif model_name == "low_dim_gauss_mix-low_dim_gauss_mix":
        step_size = 2.95e-02
    elif model_name == "gp_pois_regr-gp_pois_regr":
        step_size = 5.64e-01
    elif model_name == "mesquite-mesquite":
        step_size = 3.28e-04
    else:
        step_size = 1.0e-2

    return step_size


def get_samples(
    random_seed: int,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    pdb_toolbox = PosteriorDBToolbox(posteriordb_path)
    gs = pdb_toolbox.get_gold_standard(model_name)

    sample_dim = gs.shape[1]
    initial_sample = gs[0]
    initial_step_size = np.array([output_initial_step_size(model_name)])
    initial_covariance = np.cov(gs, rowvar=False)
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
        gs, predicted_sample = get_samples(random_number)
        median_lengthscale = Toolbox.median_trick(gs)
        mmd = Toolbox.calculate_mmd(gs, predicted_sample, median_lengthscale)
        write_results(random_number, mmd, file_path)


if __name__ == "__main__":
    main()
