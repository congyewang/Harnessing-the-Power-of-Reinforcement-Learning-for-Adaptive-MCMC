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
        "arK-arK": 1.4585439360203591,
        "mesquite-logmesquite_logvash": 1.3060794113381227,
        "earnings-logearn_interaction_z": 1.6861196334169768,
        "sblri-blr": 1.4723059041154656,
        "garch-garch11": 0.9192814322644454,
        "diamonds-diamonds": 0.9305593090862834,
        "nes1996-nes": 1.3123712218717887,
        "kidiq-kidscore_momhsiq": 1.8004793478917995,
        "low_dim_gauss_mix-low_dim_gauss_mix": 1.656372539306426,
        "kidiq-kidscore_momiq": 2.0344597692837603,
        "eight_schools-eight_schools_noncentered": 0.0319411328108101,
        "mcycle_gp-accel_gp": 0.00492445720082105,
        "hmm_example-hmm_example": 1.6673405866235762,
        "earnings-logearn_logheight_male": 1.8101872880977699,
        "nes1980-nes": 1.2838256300974658,
        "kidiq-kidscore_momhs": 2.046293062369786,
        "arma-arma11": 1.7448203409071206,
        "earnings-logearn_interaction": 1.6640156277087146,
        "eight_schools-eight_schools_centered": 0.11881713053953455,
        "hudson_lynx_hare-lotka_volterra": 1.0644198340978521,
        "kidiq_with_mom_work-kidscore_mom_work": 1.6603669530377132,
        "gp_pois_regr-gp_regr": 2.0316139654020784,
        "earnings-log10earn_height": 2.0259334027978984,
        "kidiq_with_mom_work-kidscore_interaction_z": 1.6610312326659964,
        "nes1984-nes": 1.2912750408861147,
        "nes1988-nes": 1.2886950716335848,
        "nes1972-nes": 1.3011283037649513,
        "mesquite-logmesquite": 1.2411493109216718,
        "nes1976-nes": 1.295678263975894,
        "sblrc-blr": 1.501152571430175,
        "mesquite-logmesquite_logvas": 1.2539059427738155,
        "kidiq_with_mom_work-kidscore_interaction_c": 1.6800610533118199,
        "mesquite-logmesquite_logvolume": 1.85341787398032,
        "mesquite-logmesquite_logva": 1.4951836377115655,
        "earnings-logearn_height": 2.036901934595459,
        "kidiq_with_mom_work-kidscore_interaction_c2": 1.6577074506357439,
        "kidiq-kidscore_interaction": 1.6623475565105363,
        "nes2000-nes": 1.2677328326500032,
        "bball_drive_event_1-hmm_drive_1": 1.3865659267836152,
        "mesquite-mesquite": 1.2347085298452791,
        "one_comp_mm_elim_abs-one_comp_mm_elim_abs": 0.7349372291097396,
        "nes1992-nes": 1.3121021346338757,
        "earnings-logearn_height_male": 1.8331391760913212,
        "bball_drive_event_0-hmm_drive_0": 1.2675002248981289,
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
