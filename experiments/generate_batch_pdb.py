import os
import shutil
from itertools import product
from typing import Dict

import jinja2

from pyrlmala.utils import Toolbox
from pyrlmala.utils.posteriordb import PosteriorDBToolbox


def create_model_result_dir(model_name: str) -> str:
    """
    Create a directory for the model result if it doesn't exist.

    Args:
        model_name (str): The name of the model.

    Returns:
        str: The path to the model result directory.
    """
    model_result_dir = f"./whole_results/{model_name}"
    if not os.path.exists("./whole_results"):
        os.makedirs("./whole_results")
    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)
    if not os.path.exists(f"{model_result_dir}/config"):
        os.makedirs(f"{model_result_dir}/config")

    return model_result_dir


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


def generate_files(
    model_name: str,
    repeat_num: int,
    template_root_dir: str = "./template",
    posteriordb_path: str = "../../posteriordb/posterior_database",
) -> None:
    """
    Generate the config files for the model.

    Args:
        model_name (str): The name of the model.
        repeat_num (int): The number of times to repeat the experiment.
        template_root_dir (str): The root directory for the template files.
        posteriordb_path (str): The path to the posterior database.
    """
    model_result_dir = create_model_result_dir(model_name)

    rl_algorithm_list = ["ddpg"]
    mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]
    exp_name_dict = {
        "mala": "RLMALA",
        "mala_esjd": "RLMALAESJD",
        "barker": "RLBarker",
        "barker_esjd": "RLBarkerESJD",
    }
    mcmc_env_dict = {
        "mala": "MALAEnv-v1.0",
        "mala_esjd": "MALAESJDEnv-v1.0",
        "barker": "BarkerEnv-v1.0",
        "barker_esjd": "BarkerESJDEnv-v1.0",
    }

    for rl_algorithm, mcmc_env in product(rl_algorithm_list, mcmc_env_list):
        for random_seed in range(repeat_num):
            hyperparameter_context: Dict[str, str] = {
                "exp_name": exp_name_dict[mcmc_env],
                "random_seed": str(random_seed),
                "env_id": mcmc_env_dict[mcmc_env],
                "actor_learning_rate": "1e-5",
                "exploration_noise": str(0.2 * output_initial_step_size(model_name)),
            }
            hyperparameter_template_path = (
                f"{template_root_dir}/config_template_pdb.toml"
            )

            with open(hyperparameter_template_path, "r") as file:
                hyperparameter_template = jinja2.Template(file.read())

            config_content = hyperparameter_template.render(hyperparameter_context)

            hyperparameter_config_path = f"{model_result_dir}/config/{rl_algorithm}_{mcmc_env}/{rl_algorithm}_{mcmc_env}_seed_{random_seed}.toml"
            Toolbox.create_folder(hyperparameter_config_path)
            with open(
                hyperparameter_config_path,
                "w",
            ) as file:
                file.write(config_content)

        shutil.copy(f"{template_root_dir}/actor.toml", f"{model_result_dir}/config")
        shutil.copy(f"{template_root_dir}/critic.toml", f"{model_result_dir}/config")

        pdb_run_path = f"{template_root_dir}/template_pdb_run.py"
        with open(pdb_run_path, "r") as file:
            pdb_run_template = jinja2.Template(file.read())
        pdb_run_context = {
            "model_name": model_name,
            "posteriordb_path": posteriordb_path,
            "rl_algorithm": rl_algorithm,
            "mcmc_env": mcmc_env,
            "repeat_num": repeat_num,
        }
        pdb_run_content = pdb_run_template.render(pdb_run_context)
        pdb_run_path = f"{model_result_dir}/run_pdb_{rl_algorithm}_{mcmc_env}.py"
        with open(pdb_run_path, "w") as file:
            file.write(pdb_run_content)

        bash_template_path = f"{template_root_dir}/template.run-pdb.sh"
        bash_context = {
            "model_name": model_name,
            "rl_algorithm": rl_algorithm,
            "mcmc_env": mcmc_env,
        }
        with open(bash_template_path, "r") as file:
            bash_template = jinja2.Template(file.read())
        bash_content = bash_template.render(bash_context)
        bash_script_path = f"{model_result_dir}/run_bash_{rl_algorithm}_{mcmc_env}.sh"
        with open(bash_script_path, "w") as file:
            file.write(bash_content)


if __name__ == "__main__":
    posteriordb_path = "./posteriordb/posterior_database"

    pdb_toolbox = PosteriorDBToolbox(posteriordb_path)
    model_name_list = pdb_toolbox.get_model_name_with_gold_standard()

    for model_name in model_name_list:
        posteriordb_path = "../../posteriordb/posterior_database"
        repeat_num = 10
        template_root_dir = "./template"
        generate_files(
            model_name=model_name,
            repeat_num=repeat_num,
            template_root_dir=template_root_dir,
            posteriordb_path=posteriordb_path,
        )
