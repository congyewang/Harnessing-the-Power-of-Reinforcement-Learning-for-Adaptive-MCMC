import glob
import re
from collections import defaultdict
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import typing as npt
from tqdm.auto import tqdm


class CleanPipeLine:
    @staticmethod
    def delete_duplicated_lines(input_file: str, output_file: str) -> None:
        pd.read_csv(input_file).drop_duplicates().to_csv(output_file, index=False)

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

    @classmethod
    def pipline(cls, input_file: str, output_file: str) -> None:
        cls.delete_duplicated_lines(input_file, output_file)
        cls.delete_matrics(output_file, output_file)
        cls.sort_by_random_seed(output_file, output_file)


class FlexPipeLine:
    def __init__(self, input_file: str) -> None:
        self.input_file = input_file
        self.clean_file()
        self.df = pd.read_csv(self.input_file)

    def add_header(self, header: List[str], input_file: str, output_file: str) -> None:
        with open(input_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip().split(",")
        try:
            _ = [float(item) for item in first_line]
            has_header = False
        except ValueError:
            has_header = True

        if has_header:
            pass
        else:
            df = pd.read_csv(input_file, header=None)
            df.columns = header
            df.to_csv(output_file, index=False)

    def clean_file(self) -> None:
        CleanPipeLine.delete_duplicated_lines(self.input_file, self.input_file)
        CleanPipeLine.delete_matrics(self.input_file, self.input_file)
        self.add_header(
            header=["random_seed", "mmd"],
            input_file=self.input_file,
            output_file=self.input_file,
        )

    @property
    def mean(self) -> float:
        return self.df["mmd"].mean().item()

    @property
    def median(self) -> float:
        return self.df["mmd"].median().item()

    @property
    def se(self) -> float:
        return float(self.df["mmd"].std(ddof=1) / (self.df["mmd"].count() ** 0.5))


class PlotPipeLine:
    def __init__(self) -> None:
        self.res = defaultdict(dict)
        _, self.ax = self.make_axes()

    def store_to_dict(self, file_path: str) -> None:
        df = pd.read_csv(file_path)

        mcmc_env = df["mcmc_env"].unique().item()
        step_size = df["step_size"].unique().item()

        median = df["mmd"].median()
        se = df["mmd"].std(ddof=1) / (df["mmd"].count() ** 0.5)

        self.res[mcmc_env][step_size] = {"median": median, "se": se}

    def make_axes(self) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Step Size")
        ax.set_ylabel("MMD")
        ax.set_title("Step Size vs MMD")

        return fig, ax

    def plot_const(self, mcmc_env: str = "mala") -> None:
        x_ranges = np.array(sorted([i for i in self.res[mcmc_env].keys()]))
        y_median = np.array([self.res[mcmc_env][float(x)]["median"] for x in x_ranges])
        y_se = np.array([self.res[mcmc_env][float(x)]["se"] for x in x_ranges])

        self.ax.plot(x_ranges, y_median, label=f"{mcmc_env} median")
        self.ax.fill_between(x_ranges, y_median - y_se, y_median + y_se, alpha=0.3)

        self.ax.relim()
        self.ax.autoscale_view()

        self.x_ranges = x_ranges

    def plot_flex(
        self,
        matrics: float | npt.NDArray[np.floating],
        se: float | npt.NDArray[np.floating],
    ) -> None:
        self.ax.axhline(matrics, color="red", linestyle="--", label="baseline")
        self.ax.fill_between(
            self.x_ranges, matrics - se, matrics + se, alpha=0.2, color="red"
        )

    def plot_total(
        self,
        mcmc_env: str,
        matrics: float | npt.NDArray[np.floating],
        se: float | npt.NDArray[np.floating],
        save_path: Optional[str] = None,
    ) -> None:
        self.plot_const(mcmc_env=mcmc_env)
        self.plot_flex(matrics=matrics, se=se)

        self.ax.set_xlabel("Step Size")
        self.ax.set_ylabel("MMD")
        self.ax.set_title(f"Step Size vs MMD for {mcmc_env}")

        self.ax.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def execute(
        self,
        mcmc_env: str,
        const_dir: str,
        flex_file_path: str,
        save_path: Optional[str] = None,
    ) -> None:
        file_path_list = sorted(glob.glob(f"{const_dir}/*.csv"))

        for file_path in tqdm(file_path_list):
            CleanPipeLine.delete_duplicated_lines(file_path, file_path)
            CleanPipeLine.delete_matrics(file_path, file_path)
            CleanPipeLine.sort_by_random_seed(file_path, file_path)

            self.store_to_dict(file_path)

        flex_pipeline = FlexPipeLine(input_file=flex_file_path)
        self.plot_total(
            mcmc_env=mcmc_env,
            matrics=flex_pipeline.median,
            se=flex_pipeline.se,
            save_path=save_path,
        )
