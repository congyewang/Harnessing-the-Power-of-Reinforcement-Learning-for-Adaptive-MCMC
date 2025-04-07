import re
import pandas as pd
import glob
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


class PlotToolbox:
    @staticmethod
    def delete_duplicated_lines(input_file: str, output_file: str) -> None:
        pd.read_csv(input_file).drop_duplicates().to_csv(
            output_file, index=False
        )

    @staticmethod
    def delete_matrics(input_file: str, output_file: str) -> None:
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()

        filtered_lines = [
            line for line in lines if not re.match(r"^(Mean:|Median:|SE:)", line)
        ]

        with open(output_file, "w", encoding="utf-8") as fout:
            fout.writelines(filtered_lines)

    @staticmethod
    def sort_by_random_seed(input_file: str, output_file: str) -> None:
        pd.read_csv(input_file).sort_values(by="random_seed").to_csv(
            output_file, index=False
        )


class PlotPipeLine:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.file_path_list = sorted(glob.glob("./const/*.csv"))
        self.res = defaultdict(dict)

    def store_to_dict(self, file_path: str) -> None:
        df = pd.read_csv(file_path)

        mcmc_env = df["mcmc_env"].unique().item()
        step_size = df["step_size"].unique().item()

        median = df["mmd"].median()
        se = df["mmd"].std(ddof=1) / (df["mmd"].count() ** 0.5)

        self.res[mcmc_env][step_size] = {"median": median, "se": se}

    def plot(self, mcmc_env: str = "mala"):
        x_ranges = np.array(sorted([i for i in self.res[mcmc_env].keys()]))
        y_median = np.array(
            [self.res[mcmc_env][float(x_ranges[i])]["median"] for i in range(len(x_ranges))]
        )
        y_se = np.array(
            [self.res[mcmc_env][float(x_ranges[i])]["se"] for i in range(len(x_ranges))]
        )

        plt.plot(x_ranges, y_median)
        plt.fill_between(x_ranges, y_median - y_se, y_median + y_se, alpha=0.3)
        plt.xlabel("Step Size")
        plt.ylabel("MMD")
        plt.title(f"Step Size vs MMD for {mcmc_env}")
        plt.show()

    def execute(self, mcmc_env: str = "mala"):
        for file_path in tqdm(self.file_path_list):
            PlotToolbox.delete_duplicated_lines(file_path, file_path)
            PlotToolbox.delete_matrics(file_path, file_path)
            PlotToolbox.sort_by_random_seed(file_path, file_path)

            self.store_to_dict(file_path)

        self.plot(mcmc_env)
