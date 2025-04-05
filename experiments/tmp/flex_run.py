from pyrlmala.utils.flex import FlexibleBatchRunner


def main(model_name: str) -> None:
    flex_batch_run = FlexibleBatchRunner(
        model_name=model_name,
        posteriordb_path="../posteriordb/posterior_database",
        load_policy="ensemble",
    )
    mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]

    for i in mcmc_env_list:
        flex_batch_run.run(
            mcmc_env=i,
            step_size=0.1,
            repeat_count=10,
            save_root_path=".",
            template_path="./config/config_template.toml",
            output_root_path="./config",
        )


if __name__ == "__main__":
    model_name = "test-laplace_1-test-laplace_1"
    main(model_name)
