import numpy as np
from numpy import typing as npt
from tqdm.auto import tqdm

from pyrlmala.learning import LearningFactory
from pyrlmala.utils import CalculateMMD, Toolbox

model_name = "test-laplace_4-test-laplace_4"
posteriordb_path = "../posteriordb/posterior_database"


def run(
    random_seed: int, gs: npt.NDArray[np.float64], step_size: float, mcmc_env: str
) -> float:
    sample_dim = 2
    initial_sample = 0.1 * np.ones(sample_dim)
    initial_step_size = np.array([step_size])
    algorithm = "ddpg"

    learning_instance = LearningFactory.create_learning_instance(
        algorithm=algorithm,
        model_name=model_name,
        posteriordb_path=posteriordb_path,
        initial_sample=initial_sample,
        initial_step_size=initial_step_size,
        hyperparameter_config_path=f"./config/{algorithm}_{mcmc_env}/{algorithm}_{mcmc_env}_seed_{random_seed}.toml",
        actor_config_path="./config/actor.toml",
        critic_config_path="./config/critic.toml",
        verbose=False,
    )

    learning_instance.train()

    # Calculate MMD
    accepted_sample = learning_instance.env.get_attr("store_accepted_sample")[0][
        -len(gs) :
    ]
    mmd = CalculateMMD.calculate(gs, accepted_sample)

    return mmd


def write_results(random_seed: int, mmd: float, save_file_path: str) -> None:
    with open(save_file_path, "a+") as f:
        f.write(f"{random_seed}, {mmd}\n")


def main() -> None:
    mcmc_env = "mala"
    step_size = 0.5
    save_file_path = f"{model_name}_{mcmc_env}_mmd.txt"
    random_seeds = range(10)
    gs = Toolbox.gold_standard(model_name, posteriordb_path)

    mmd_res = np.empty(len(random_seeds))

    for i in tqdm(random_seeds):
        try:
            mmd = run(i, gs, step_size, mcmc_env)
            mmd_res[i] = mmd
            write_results(i, mmd, save_file_path)
        except Exception as e:
            print(f"Error in seed {i}: {e}")

    with open(save_file_path, "a+") as f:
        f.write(f"Mean: {mmd_res.mean()}\n")
        f.write(f"SE: {mmd_res.std(ddof=1) / np.sqrt(len(mmd_res))}\n")


if __name__ == "__main__":
    main()
