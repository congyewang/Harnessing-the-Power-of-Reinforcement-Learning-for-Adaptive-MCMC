import glob
import re
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from prettytable import PrettyTable, TableStyle
from tqdm.auto import tqdm

from pyrlmala.utils.export import PosteriorDBToolbox
from pyrlmala.utils.read import ResultReader

warnings.filterwarnings(
    "ignore",
    message="Loading a shared object .* that has already been loaded",
    category=UserWarning,
    module="bridgestan.model",
)
RESULTS_DIR = "./whole_results_time"
posteriordb_path = "./posteriordb/posterior_database"


class CPUTimeReader(ResultReader):
    def __init__(
        self, results_dir: str, repeat_num: int = 10, method_name: Optional[str] = None
    ) -> None:
        """
        Initialize the CPUTimeReader class.

        Args:
            results_dir (str): Directory where CPU time result CSV files are located.
            repeat_num (int): Number of repetitions for loading results.
        """
        super().__init__(results_dir, repeat_num)
        self.method_name = method_name

    def load_results(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Parse all results into structured dict.

        Returns:
            Dict[str, Dict[str, List[float]]]: {model_name: {method: [median, q1, q3]}}
        """
        mcmc_env_pattern = r"(mala(?:_[a-zA-Z0-9]+)?|barker(?:_[a-zA-Z0-9]+)?)"
        csv_path_list = glob.glob(f"{self.results_dir}/**/*.csv", recursive=True)

        results: Dict[str, Dict[str, List[float]]] = defaultdict(dict)

        for path in tqdm(csv_path_list):
            parts = path.split("/")
            if len(parts) < 3:
                continue
            model_name = parts[-2]
            filename = parts[-1]

            if self.method_name:
                method = self.method_name
            else:
                mcmc_env_match = re.search(mcmc_env_pattern, filename)
                if not mcmc_env_match:
                    logger.warning(f"Skip {filename}: no match for method.")
                    continue
                method = mcmc_env_match.group()

            df = pd.read_csv(path)
            if df.shape[0] < 1:
                logger.warning(
                    f"Skipping {model_name} {method}: incomplete result ({df.shape[0]} rows)"
                )
                continue

            cpu_time_values = df["cpu_time"].values

            median = float(np.median(cpu_time_values))
            q1 = float(np.percentile(cpu_time_values, 25))
            q3 = float(np.percentile(cpu_time_values, 75))

            mean = float(np.mean(cpu_time_values))
            se = float(np.std(cpu_time_values) / np.sqrt(self.repeat_num))

            results[model_name][method] = [median, q1, q3, mean, se]

        return results


class CPUTimeGenerator:
    def __init__(
        self,
        result_reader: ResultReader,
        posteriordb_path: str,
        output_path: str = "cpu_time_results.md",
    ) -> None:
        """
        Initialize the PosteriorDBGenerator class.

        Args:
            result_reader (ResultReader): An instance of a ResultReader class to read results.
            posteriordb_path (str): Path to the PosteriorDB directory.
            output_path (str): Path for the output markdown file.
        """
        self.result_reader = result_reader
        self.posteriordb_path = posteriordb_path
        self.pdb_toolbox = PosteriorDBToolbox(posteriordb_path)
        self.cpu_time_results = self.result_reader.load_results()
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
        Write the CPU time results to a markdown file in a table format, including model dimensions.
        """
        res_sorted = self.get_sorted_model_names()

        # Gather all method keys
        all_keys = set()
        for model_dict in self.cpu_time_results.values():
            all_keys.update(model_dict.keys())
        all_keys = sorted(all_keys)

        # Add Dimension column after Model
        field_names = ["Model", "d"]
        for key in all_keys:
            field_names.extend(
                [f"{key} Median", f"{key} Q1", f"{key} Q3", f"{key} Mean", f"{key} SE"]
            )

        table = PrettyTable()
        table.set_style(TableStyle.MARKDOWN)
        table.field_names = field_names

        for model_dim, model_name in res_sorted:
            cpu_time_dict = self.cpu_time_results.get(model_name, {})
            row = [model_name, model_dim]
            for key in all_keys:
                values = cpu_time_dict.get(key)
                if values and len(values) >= 5:
                    row.extend([f"{v:.4g}" for v in values[:5]])
                else:
                    row.extend(["-"] * 5)
            table.add_row(row)

        with open(self.output_path, "w") as f:
            f.write(table.get_string())

    def get_result_dataframe(self) -> pd.DataFrame:
        """
        Return the CPU time results as a pandas DataFrame, including model dimensions.

        Returns:
            pd.DataFrame: A DataFrame containing model names, dimensions, and CPU time statistics.
        """
        res_sorted = self.get_sorted_model_names()

        # Collect all available method keys
        all_keys = set()
        for model_dict in self.cpu_time_results.values():
            all_keys.update(model_dict.keys())
        all_keys = sorted(all_keys)

        # Set up field names with Model and Dimension
        field_names = ["Model", "d"]
        for key in all_keys:
            field_names.extend(
                [f"{key} Median", f"{key} Q1", f"{key} Q3", f"{key} Mean", f"{key} SE"]
            )

        # Construct rows
        rows = []
        for model_dim, model_name in res_sorted:
            cpu_time_dict = self.cpu_time_results.get(model_name, {})
            row = [model_name, model_dim]  # include dimension here
            for key in all_keys:
                values = cpu_time_dict.get(key)
                if values and len(values) >= 5:
                    row.extend(values[:5])
                else:
                    row.extend([None] * 5)
            rows.append(row)

        return pd.DataFrame(rows, columns=field_names)

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

                if df.shape[0] != self.result_reader.repeat_num:
                    sh_command = f"cd {i.split('/')[2]}\nsbatch run_bash_{re.search('ddpg.+', i.split('/')[3]).group().replace('.csv', '')}.sh\ncd -\n\n"
                    f.write(sh_command)

    def execute(self, mode: str = "pandas") -> None | pd.DataFrame:
        """
        Execute the main functionality of the PosteriorDBGenerator class.

        Args:
            mode (str): Execution mode - "pandas" for DataFrame output, "markdown" for file output.

        Returns:
            None | pd.DataFrame: Returns DataFrame if mode is "pandas", otherwise None.
        """
        match mode:
            case "pandas":
                return self.get_result_dataframe()
            case "markdown":
                self.write_result_to_markdown()
            case _:
                self.write_result_to_markdown()


def generate_markdown(mode: str) -> None:
    """
    Generate a markdown file with the results of the experiments.
    """
    match mode:
        case "cpu_time":
            cpu_time_result_reader = CPUTimeReader(results_dir=f"{RESULTS_DIR}")
            CPUTimeGenerator(
                cpu_time_result_reader,
                posteriordb_path,
                output_path="cpu_time_results.md",
            ).execute()
        case _:
            raise ValueError(f"Unknown mode: {mode}. Use 'cpu_time'.")


def main() -> None:
    """
    Main function to generate markdown files for baseline and flexible MCMC results.
    """
    generate_markdown("cpu_time")


if __name__ == "__main__":
    main()
