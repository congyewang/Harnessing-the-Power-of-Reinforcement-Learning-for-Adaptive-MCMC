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
    step_size_dict = {
        "arK-arK": 1.444885116153801,
        "mesquite-logmesquite_logvash": 1.3026946593080484,
        "earnings-logearn_interaction_z": 1.6773751049582213,
        "sblri-blr": 1.4871071073090385,
        "garch-garch11": 0.9281432387660594,
        "diamonds-diamonds": 0.9227758660447334,
        "nes1996-nes": 1.2985494241661342,
        "kidiq-kidscore_momhsiq": 1.8174460275661013,
        "low_dim_gauss_mix-low_dim_gauss_mix": 1.6616931193623956,
        "kidiq-kidscore_momiq": 2.052443006288145,
        "eight_schools-eight_schools_noncentered": 0.032297853500002846,
        "mcycle_gp-accel_gp": 0.00599514936951625,
        "hmm_example-hmm_example": 1.6537297541905724,
        "earnings-logearn_logheight_male": 1.837545169809095,
        "nes1980-nes": 1.265225407271176,
        "kidiq-kidscore_momhs": 2.0759726980979143,
        "arma-arma11": 1.7406216402946875,
        "earnings-logearn_interaction": 1.651080354596432,
        "eight_schools-eight_schools_centered": 0.11047533827053524,
        "hudson_lynx_hare-lotka_volterra": 1.0764082604774006,
        "kidiq_with_mom_work-kidscore_mom_work": 1.644490013581835,
        "gp_pois_regr-gp_regr": 2.025951555179607,
        "earnings-log10earn_height": 2.0627303833602317,
        "kidiq_with_mom_work-kidscore_interaction_z": 1.6633501545611526,
        "nes1984-nes": 1.2669750817391519,
        "nes1988-nes": 1.3105343227352024,
        "nes1972-nes": 1.2733228411882598,
        "mesquite-logmesquite": 1.2337267837763584,
        "nes1976-nes": 1.2812554197886838,
        "sblrc-blr": 1.4868078422514857,
        "mesquite-logmesquite_logvas": 1.2282993053337004,
        "kidiq_with_mom_work-kidscore_interaction_c": 1.641862772269111,
        "mesquite-logmesquite_logvolume": 1.872042337549332,
        "mesquite-logmesquite_logva": 1.4817593533186844,
        "earnings-logearn_height": 2.0349023121888217,
        "kidiq_with_mom_work-kidscore_interaction_c2": 1.6219512406858225,
        "kidiq-kidscore_interaction": 1.667698269421378,
        "nes2000-nes": 1.2835527435404193,
        "bball_drive_event_1-hmm_drive_1": 1.39602670527713,
        "mesquite-mesquite": 1.2386738530403414,
        "one_comp_mm_elim_abs-one_comp_mm_elim_abs": 0.7417336474220781,
        "nes1992-nes": 1.2956799224394653,
        "earnings-logearn_height_male": 1.8523044897309613,
        "bball_drive_event_0-hmm_drive_0": 1.2723081527243076,
        "earnings-earn_height": 1.7e-5,
        "kilpisjarvi_mod-kilpisjarvi": 3.75e-2,
        "gp_pois_regr-gp_pois_regr": 5.64e-1,
    }

    step_size = step_size_dict.get(model_name, 1e-2)

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
