from pyrlmala.utils.plot import PlotPipeLine


model_list = ["test-banana-test-banana", "test-neals_funnel-test-neals_funnel"] + [
    f"test-laplace_{i}-test-laplace_{i}" for i in [1, 2, 4]
]

# mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]
mcmc_env_list = ["mala"]

bootstrap_res = {
    "test-banana-test-banana": (
        6.693579028532737e-05,
        3.252257778818546e-05,
        9.941779793612437e-05,
    ),
    "test-neals_funnel-test-neals_funnel": (
        6.790319804189648e-05,
        5.552418464618847e-05,
        9.22387055656837e-05,
    ),
    "test-laplace_1-test-laplace_1": (
        6.880791394087149e-05,
        5.886874207775006e-05,
        8.673648262561007e-05,
    ),
    "test-laplace_2-test-laplace_2": (
        0.00010108379915718668,
        5.5503480059976296e-05,
        0.00012444663540062129,
    ),
    "test-laplace_4-test-laplace_4": (
        5.535868116945952e-05,
        4.547333266832765e-05,
        8.727203370076375e-05,
    ),
}

for model_name in model_list:
    model_dir_root_name = model_name.split("-")[1]
    const_dir = f"./{model_dir_root_name}/const"
    for mcmc_env in mcmc_env_list:
        PlotPipeLine().execute(
            mcmc_env=mcmc_env,
            const_dir=const_dir,
            flex_file_path=f"./{model_dir_root_name}/flex/{model_name}_{mcmc_env}_mmd.txt",
            bootstrap_tuple=bootstrap_res[model_name],
            save_path=f"./{model_dir_root_name}/pic/{model_name}_{mcmc_env}_mmd.pdf",
        )
