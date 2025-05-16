import matplotlib.pyplot as plt

from pyrlmala.utils.plot import PlotPipeLine

model_list = [
    "test-laplace_1-test-laplace_1",
    "test-laplace_2-test-laplace_2",
    "test-neals_funnel-test-neals_funnel",
    "test-banana-test-banana",
    "test-skew_t-test-skew_t",
]

mcmc_env_list = ["mala"]


fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)

for idx, model_name in enumerate(model_list):
    model_dir_root_name = model_name.split("-")[1]
    const_dir = f"./{model_dir_root_name}/const"
    for mcmc_env in mcmc_env_list:
        PlotPipeLine(axes=axes[idx]).execute(
            mcmc_env=mcmc_env,
            const_dir=const_dir,
            flex_file_path=f"./{model_dir_root_name}/flex/{model_name}_{mcmc_env}_mmd.txt",
            save_path=f"./{model_dir_root_name}/pic/{model_name}_{mcmc_env}_mmd.pdf",
        )
