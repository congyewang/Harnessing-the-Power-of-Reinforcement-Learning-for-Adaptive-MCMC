import copy

import numpy as np
import torch

from pyrlmala.learning import LearningFactory
from pyrlmala.utils.plot import PolicyPlot


def run_experiment(model_name: str, mcmc_env: str):
    posteriordb_path = "./posteriordb/posterior_database"
    model_dir_root_name = model_name.split("-")[1]

    sample_dim = 2
    initial_sample = 0.1 * np.ones(sample_dim)
    initial_step_size = np.array([0.1])
    algorithm = "ddpg"

    learning_instance = LearningFactory.create_learning_instance(
        algorithm=algorithm,
        model_name=model_name,
        posteriordb_path=posteriordb_path,
        initial_sample=initial_sample,
        initial_step_size=initial_step_size,
        hyperparameter_config_path=f"./{model_dir_root_name}/config/{algorithm}_{mcmc_env}.toml",
        actor_config_path=f"./{model_dir_root_name}/config/actor.toml",
        critic_config_path=f"./{model_dir_root_name}/config/critic.toml",
    )

    learning_instance.train()

    best_actor = copy.deepcopy(learning_instance.actor)
    best_actor.load_state_dict(learning_instance.topk_policy.topk()[0][1]["actor"])
    best_policy = lambda x: best_actor(x.double())

    x_range = (-3, 3, 0.1)
    y_range = (-3, 3, 0.1)
    PolicyPlot.policy_plot_2D_heatmap(
        best_policy,
        torch.arange(*x_range),
        torch.arange(*y_range),
        save_path=f"./{model_dir_root_name}/pic/{model_name}_{algorithm}_{mcmc_env}_best_policy.pdf",
    )


def main() -> None:
    model_list = [f"test-laplace_{i}-test-laplace_{i}" for i in [1, 2, 4]]
    mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]

    for model_name in model_list:
        for mcmc_env in mcmc_env_list:
            run_experiment(model_name, mcmc_env)


if __name__ == "__main__":
    main()
