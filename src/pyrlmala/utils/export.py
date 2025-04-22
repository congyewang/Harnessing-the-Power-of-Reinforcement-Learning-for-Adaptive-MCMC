import glob
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .posteriordb import PosteriorDBToolbox


class PosteriorDBGenerator:
    def __init__(self, results_dir: str, posteriordb_path: str) -> None:
        """
        Initialize the PosteriorDBGenerator class.

        Args:
            results_dir (str): Directory containing the results.
            posteriordb_path (str): Path to the posterior database.
        """
        self.results_dir = results_dir

        self.posteriordb_path = posteriordb_path
        self.pdb_toolbox = PosteriorDBToolbox(posteriordb_path)

        self.mmd_results = self.export_result_to_dict()

    def export_result_to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Export the results to a dictionary.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing the results.
        """
        s = 0
        csv_path_list = glob.glob(f"./{self.results_dir}/*/*.csv")
        mcmc_env_pattern = r"(mala(?:_[a-zA-Z0-9]+)?|barker(?:_[a-zA-Z0-9]+)?)"
        mmd_results: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for path in tqdm(csv_path_list):
            model_name = path.split("/")[2]
            csv_file_name = path.split("/")[3]
            mcmc_env = re.search(mcmc_env_pattern, csv_file_name).group()
            df = pd.read_csv(path)

            if df.shape[0] != 10:
                s += 1
                print(f"Error in {model_name} {mcmc_env} {df.shape[0]}")
                continue
            else:
                mmd_results[model_name][mcmc_env] = [
                    df["mmd"].median(),
                    np.percentile(df["mmd"], 25),
                    np.percentile(df["mmd"], 75),
                ]

        return mmd_results

    def get_sorted_model_names(self) -> List[Tuple[int, str]]:
        """
        Get the sorted model names based on the number of parameters in the gold standard.

        Returns:
            List[Tuple[int, str]]: A list of tuples containing the number of parameters and the model name, sorted by the number of parameters.
        """
        res: List[Tuple[int, str]] = []
        for model_name in tqdm(self.pdb_toolbox.get_model_name_with_gold_standard()):
            if "test" not in model_name:
                res.append(
                    (
                        self.pdb_toolbox.get_gold_standard(model_name).shape[1],
                        model_name,
                    )
                )

        res_sorted = sorted(res)
        return res_sorted

    def write_result_to_markdown(self) -> None:
        """
        Write the results to a markdown file.
        """
        res_sorted = self.get_sorted_model_names()

        # Write the results to a markdown file
        with open("mmd_results.md", "w") as f:
            f.write(
                "| Model | MALA Median | MALA Q1 | MALA Q3| MALA ESJD Median | MALA ESJD Q1 | MALA ESJD Q3 | Barker Median | Barker Q1 | Barker Q3 | Barker ESJD Median | Barker ESJD Q1 | Barker ESJD Q3 |\n"
            )
            f.write(
                "|-------|-------------|---------|--------|------------------|--------------|--------------|---------------|----------|--------|--------------------|----------------|----------------|\n"
            )
            for _, model_name in res_sorted:
                mmd_dict = self.mmd_results.get(model_name, {})
                row = [model_name]
                for key in ["mala", "mala_esjd", "barker", "barker_esjd"]:
                    values = mmd_dict.get(key, None)
                    if values and len(values) >= 3:
                        formatted = [f"{v:.4g}" for v in values[:3]]
                        row.extend(formatted)
                    else:
                        row.extend(["-"] * 3)
                f.write("| " + " | ".join(row) + " |\n")

    def export_failed_bash(self) -> None:
        """
        Export bash commands for failed runs.
        """
        csv_path_list = glob.glob(f"./{self.results_dir}/*/*.csv")

        with open("submit_failed.sh", "w") as f:
            for i in tqdm(csv_path_list):
                df = pd.read_csv(i)

                if df.shape[0] != 10:
                    sh_command = f"cd {i.split("/")[2]}\nsbatch run_bash_{re.search("ddpg.+", i.split("/")[3]).group().replace(".csv", "")}.sh\ncd -\n\n"
                    f.write(sh_command)

    def execute(self) -> None:
        """
        Execute the main functionality of the PosteriorDBGenerator class.
        """
        self.write_result_to_markdown()
