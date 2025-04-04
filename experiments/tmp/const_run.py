from tqdm.auto import trange

from pyrlmala.utils.benchmark import MMDBenchMark

model_name = "test-laplace_4-test-laplace_4"
posteriordb_path = "../posteriordb/posterior_database"


def run(mcmc_env: str, random_seed: int, step_size: float) -> None:
    const_mmd_instance = MMDBenchMark(
        mcmc_env=mcmc_env,
        model_name=model_name,
        posteriordb_path=posteriordb_path,
        random_seed=random_seed,
        step_size=step_size,
        verbose=False,
    )
    const_mmd_instance.execute()


def main() -> None:
    mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]
    repeat_count = 10
    step_size_list = list(range(1, 20)) + [0.1, 0.5]

    for mcmc_env in mcmc_env_list:
        for random_seed in trange(repeat_count):
            for step_size in step_size_list:
                run(mcmc_env, random_seed, step_size)


if __name__ == "__main__":
    main()
