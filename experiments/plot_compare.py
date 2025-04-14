from pyrlmala.utils.plot import PlotPipeLine


model_list = [f"test-laplace_{i}-test-laplace_{i}" for i in [1, 2, 4]]

mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]

bootstrap_res = {
    model_list[0]: (0.12712642355339562,0.12657054195328704,0.1273794465236633),
    model_list[1]: (0.10267085216084658,0.10173536978025989,0.10312433650693831),
    model_list[2]: (0.08310532977675208,0.08253760609758073,0.08366187742640738),
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
