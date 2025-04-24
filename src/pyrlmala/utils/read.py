import glob
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm


class ResultReader(ABC):
    def __init__(self, results_dir: str, repeat_num: int = 10) -> None:
        """
        Initialize the ResultReader class.

        Args:
            results_dir (str): Directory where result CSV files are located.
            repeat_num (int): Number of repetitions for loading results.
        """
        self.results_dir = results_dir
        self.repeat_num = repeat_num

    @abstractmethod
    def load_results(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Return results in the form: {model_name: {method: [median, q1, q3]}}

        Returns:
            Dict[str, Dict[str, List[float]]]: A dictionary containing the results.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaselineResultReader(ResultReader):
    def __init__(
        self, results_dir: str, repeat_num: int = 10, method_name: str = "baseline"
    ) -> None:
        """
        Initialize the BaselineResultReader class.

        Args:
            results_dir (str): Directory where baseline result CSV files are located.
            method_name (str): Method name under which results will be recorded.
            repeat_num (int): Number of repetitions for loading results.
        """
        super().__init__(results_dir, repeat_num)
        self.method_name = method_name

    def load_results(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Returns:
            Dict[str, Dict[str, List[float]]]: {model_name: {method: [median, q1, q3]}}
        """
        results: Dict[str, Dict[str, List[float]]] = defaultdict(dict)

        csv_paths = glob.glob(f"{self.results_dir}/**/*.csv", recursive=True)
        for path in csv_paths:
            df = pd.read_csv(path)

            if df.shape[0] != self.repeat_num:
                logger.warning(f"Skipping {path}: incorrect number of rows.")
                continue

            model_name = df["model_name"].iloc[0]
            mmd_values = df["mmd"].values

            median = float(np.median(mmd_values))
            q1 = float(np.percentile(mmd_values, 25))
            q3 = float(np.percentile(mmd_values, 75))

            results[model_name][self.method_name] = [median, q1, q3]

        return results


class MCMCResultReader(ResultReader):
    def __init__(self, results_dir: str, repeat_num: int = 10) -> None:
        """
        Initialize the MCMCResultReader class.

        Args:
            results_dir (str): Directory where MCMC result CSV files are located.
            repeat_num (int): Number of repetitions for loading results.
        """
        super().__init__(results_dir, repeat_num)

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

            mcmc_env_match = re.search(mcmc_env_pattern, filename)
            if not mcmc_env_match:
                print(f"[Warning] Skip {filename}: no match for method.")
                continue
            method = mcmc_env_match.group()

            df = pd.read_csv(path)
            if df.shape[0] != self.repeat_num:
                logger.warning(
                    f"Skipping {model_name} {method}: incomplete result ({df.shape[0]} rows)"
                )
                continue

            mmd_values = df["mmd"].values
            median = float(np.median(mmd_values))
            q1 = float(np.percentile(mmd_values, 25))
            q3 = float(np.percentile(mmd_values, 75))
            results[model_name][method] = [median, q1, q3]

        return results
