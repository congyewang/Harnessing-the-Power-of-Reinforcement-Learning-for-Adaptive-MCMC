import numpy as np

from pyrlmala.utils.flex import FlexibleBatchRunner

if __name__ == "__main__":
    flex_batch_run = FlexibleBatchRunner(
        model_name="test-neals_funnel-test-neals_funnel",
        posteriordb_path="../posteriordb/posterior_database",
        load_policy="ensemble",
    )
    # mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]
    mcmc_env_list = ["mala"]

    for i in mcmc_env_list:
        flex_batch_run.run(
            mcmc_env=i,
            step_size=0.1,
            initial_sample=np.array([0.0, -4.0]),
            repeat_count=10,
            save_root_path=".",
            template_path="./config/config_template.toml",
            output_root_path="./config",
        )

    import requests

    requests.post("https://ntfy.greenlimes.top/asus", data="Finished")
