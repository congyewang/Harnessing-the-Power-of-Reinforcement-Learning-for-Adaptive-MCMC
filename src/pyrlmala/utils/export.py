import glob
import re
import warnings
from typing import List, Tuple

import pandas as pd
from prettytable import PrettyTable, TableStyle
from tqdm.auto import tqdm

from .posteriordb import PosteriorDBToolbox
from .read import ResultReader

warnings.filterwarnings(
    "ignore",
    message="Loading a shared object .* that has already been loaded",
    category=UserWarning,
    module="bridgestan.model",
)


class PosteriorDBGenerator:
    def __init__(self, result_reader: ResultReader, posteriordb_path: str, output_path: str = "mmd_results.md") -> None:
        """
        Initialize the PosteriorDBGenerator class.

        Args:
            result_reader (ResultReader): An instance of a ResultReader class to read results.
            posteriordb_path (str): Path to the PosteriorDB directory.
        """
        self.result_reader = result_reader
        self.posteriordb_path = posteriordb_path
        self.pdb_toolbox = PosteriorDBToolbox(posteriordb_path)
        self.mmd_results = self.result_reader.load_results()
        self.output_path = output_path

    def get_sorted_model_names(self) -> List[Tuple[int, str]]:
        """
        Get sorted model names based on the number of parameters in the gold standard.

        Returns:
            List[Tuple[int, str]]: A list of tuples containing the number of parameters and model names.
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
        return sorted(res)

    def write_result_to_markdown(self) -> None:
        """
        Write the MMD results to a markdown file in a table format.
        """
        res_sorted = self.get_sorted_model_names()

        all_keys = set()
        for model_dict in self.mmd_results.values():
            all_keys.update(model_dict.keys())
        all_keys = sorted(all_keys)

        field_names = ["Model"]
        for key in all_keys:
            field_names.extend([f"{key} Median", f"{key} Q1", f"{key} Q3"])

        table = PrettyTable()
        table.set_style(TableStyle.MARKDOWN)
        table.field_names = field_names

        for _, model_name in res_sorted:
            mmd_dict = self.mmd_results.get(model_name, {})
            row = [model_name]
            for key in all_keys:
                values = mmd_dict.get(key)
                if values and len(values) >= 3:
                    row.extend([f"{v:.4g}" for v in values[:3]])
                else:
                    row.extend(["-"] * 3)
            table.add_row(row)

        with open(self.output_path, "w") as f:
            f.write(table.get_string())

    def export_failed_bash(self) -> None:
        """
        Export bash commands to resubmit failed jobs.
        """
        csv_path_list = glob.glob(
            f"{self.result_reader.results_dir}/**/*.csv", recursive=True
        )

        with open("submit_failed.sh", "w") as f:
            for i in tqdm(csv_path_list):
                df = pd.read_csv(i)

                if df.shape[0] != 10:
                    sh_command = f"cd {i.split('/')[2]}\nsbatch run_bash_{re.search('ddpg.+', i.split('/')[3]).group().replace('.csv', '')}.sh\ncd -\n\n"
                    f.write(sh_command)

    def execute(self) -> None:
        """
        Execute the main functionality of the PosteriorDBGenerator class.
        """
        self.write_result_to_markdown()
