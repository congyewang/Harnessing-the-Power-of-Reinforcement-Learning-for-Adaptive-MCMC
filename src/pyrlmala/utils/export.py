import glob
import io
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
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
    def __init__(
        self,
        result_reader: ResultReader,
        posteriordb_path: str,
        output_path: str = "mmd_results.md",
    ) -> None:
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
        Write the MMD results to a markdown file in a table format, including model dimensions.
        """
        res_sorted = self.get_sorted_model_names()

        # Gather all metric keys
        all_keys = set()
        for model_dict in self.mmd_results.values():
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
            mmd_dict = self.mmd_results.get(model_name, {})
            row = [model_name, model_dim]
            for key in all_keys:
                values = mmd_dict.get(key)
                if values and len(values) >= 5:
                    row.extend([f"{v:.4g}" for v in values[:5]])
                else:
                    row.extend(["-"] * 5)
            table.add_row(row)

        with open(self.output_path, "w") as f:
            f.write(table.get_string())

    def get_result_dataframe(self) -> pd.DataFrame:
        """
        Return the MMD results as a pandas DataFrame, including model dimensions.

        Returns:
            pd.DataFrame: A DataFrame containing model names, dimensions, and MMD statistics.
        """
        res_sorted = self.get_sorted_model_names()

        # Collect all available keys
        all_keys = set()
        for model_dict in self.mmd_results.values():
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
            mmd_dict = self.mmd_results.get(model_name, {})
            row = [model_name, model_dim]  # include dimension here
            for key in all_keys:
                values = mmd_dict.get(key)
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

                if df.shape[0] != 10:
                    sh_command = f"cd {i.split('/')[2]}\nsbatch run_bash_{re.search('ddpg.+', i.split('/')[3]).group().replace('.csv', '')}.sh\ncd -\n\n"
                    f.write(sh_command)

    def execute(self, mode: str = "pandas") -> None | pd.DataFrame:
        """
        Execute the main functionality of the PosteriorDBGenerator class.
        """
        match mode:
            case "pandas":
                self.get_sorted_model_names
        self.write_result_to_markdown()


class TableGenerator:
    @staticmethod
    def read_markdown_table(file_path: str) -> pd.DataFrame:
        """
        Read a markdown table from a file and convert it to a DataFrame.

        Args:
            file_path (str): Path to the markdown file.

        Returns:
            pd.DataFrame: DataFrame containing the table data.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find the line where the table starts (starts with |)
        header_index = next(i for i, line in enumerate(lines) if re.match(r"^\|", line))
        table_lines = lines[header_index:]

        return (
            pd.read_csv(io.StringIO("".join(table_lines)), sep="|", engine="python")
            .dropna(axis=1, how="all")
            .iloc[1:]
            .reset_index(drop=True)
        )

    @staticmethod
    def format_scientific_with_error(
        center: Optional[float], spread: Optional[float]
    ) -> str:
        """
        Format a number with its uncertainty in scientific notation.

        Args:
            center (Optional[float]): The central value.
            spread (Optional[float]): The spread or uncertainty.

        Returns:
            str: Formatted string in scientific notation.
        """
        if pd.isna(center) or pd.isna(spread):
            return "-"
        if center == 0:
            return "0.0(0.0)E0"
        exp = int(np.floor(np.log10(abs(center))))
        base = center / (10**exp)
        relative_spread = spread / (10**exp)
        return f"{base:.1f}({relative_spread:.1f})E{exp}"

    @classmethod
    def apply_transformation_with_highlight(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformation to the DataFrame and highlight the smallest values
        among four methods for either Mean(SE) or Mid(IQR) or both if available.
        If RMALA-RLMH CDLB and RMALA-RLMH LESJD have the same minimum value, both are highlighted.
        Output columns ordered as: RMALA AAR, RMALA ESJD, RMALA-RLMH LESJD, RMALA-RLMH CDLB.

        Args:
            df (pd.DataFrame): Input DataFrame to be transformed.

        Returns:
            pd.DataFrame: Transformed DataFrame with highlighted values.
        """
        df.columns = df.columns.str.strip()
        df = df.replace(r"\*\*(.*?)\*\*", r"\1", regex=True)
        df = df.replace(r"^\s*-+\s*$", np.nan, regex=True)
        df = df.replace(r"^\s*$", np.nan, regex=True)

        def fmt(center: float, spread: float, highlight: bool = False) -> str:
            result = cls.format_scientific_with_error(center, spread)
            return f"\\textbf{{{result}}}" if highlight else result

        has_median = all(
            col in df.columns
            for col in [
                "RMALA AAR Median",
                "RMALA ESJD Median",
                "RMALA-RLMH LESJD Median",
                "RMALA-RLMH CDLB Median",
            ]
        )
        has_mean = all(
            col in df.columns
            for col in [
                "RMALA AAR Mean",
                "RMALA ESJD Mean",
                "RMALA-RLMH LESJD Mean",
                "RMALA-RLMH CDLB Mean",
            ]
        )

        rows: List[Dict[str, str]] = []

        for _, row in df.iterrows():
            model: str = row["Model"]
            dim = int(row["d"]) if "d" in row and not pd.isna(row["d"]) else None
            out_row = {"Model": model, "d": dim}

            if has_median:
                med_rl, q1_rl, q3_rl = map(
                    float,
                    (
                        row["RMALA-RLMH CDLB Median"],
                        row["RMALA-RLMH CDLB Q1"],
                        row["RMALA-RLMH CDLB Q3"],
                    ),
                )
                med_base, q1_base, q3_base = map(
                    float,
                    (
                        row["RMALA AAR Median"],
                        row["RMALA AAR Q1"],
                        row["RMALA AAR Q3"],
                    ),
                )
                med_esjd, q1_esjd, q3_esjd = map(
                    float,
                    (
                        row["RMALA ESJD Median"],
                        row["RMALA ESJD Q1"],
                        row["RMALA ESJD Q3"],
                    ),
                )
                med_lesjd, q1_lesjd, q3_lesjd = map(
                    float,
                    (
                        row["RMALA-RLMH LESJD Median"],
                        row["RMALA-RLMH LESJD Q1"],
                        row["RMALA-RLMH LESJD Q3"],
                    ),
                )

                med_values = [med_rl, med_base, med_esjd, med_lesjd]
                min_med_value = min(med_values)

                is_rl_min = med_rl == min_med_value
                is_lesjd_min = med_lesjd == min_med_value
                is_base_min = med_base == min_med_value
                is_esjd_min = med_esjd == min_med_value

                out_row["RMALA AAR Mid(IQR)"] = fmt(
                    med_base, q3_base - q1_base, is_base_min
                )
                out_row["RMALA ESJD Mid(IQR)"] = fmt(
                    med_esjd, q3_esjd - q1_esjd, is_esjd_min
                )
                out_row["RMALA-RLMH LESJD Mid(IQR)"] = fmt(
                    med_lesjd, q3_lesjd - q1_lesjd, is_lesjd_min
                )
                out_row["RMALA-RLMH CDLB Mid(IQR)"] = fmt(
                    med_rl, q3_rl - q1_rl, is_rl_min
                )

            if has_mean:
                mean_rl, se_rl = map(
                    float, (row["RMALA-RLMH CDLB Mean"], row["RMALA-RLMH CDLB SE"])
                )
                mean_base, se_base = map(
                    float, (row["RMALA AAR Mean"], row["RMALA AAR SE"])
                )
                mean_esjd, se_esjd = map(
                    float, (row["RMALA ESJD Mean"], row["RMALA ESJD SE"])
                )
                mean_lesjd, se_lesjd = map(
                    float, (row["RMALA-RLMH LESJD Mean"], row["RMALA-RLMH LESJD SE"])
                )

                mean_values = [mean_rl, mean_base, mean_esjd, mean_lesjd]
                min_mean_value = min(mean_values)

                is_rl_min = mean_rl == min_mean_value
                is_lesjd_min = mean_lesjd == min_mean_value
                is_base_min = mean_base == min_mean_value
                is_esjd_min = mean_esjd == min_mean_value

                out_row["RMALA AAR Mean(SE)"] = fmt(mean_base, se_base, is_base_min)
                out_row["RMALA ESJD Mean(SE)"] = fmt(mean_esjd, se_esjd, is_esjd_min)
                out_row["RMALA-RLMH LESJD Mean(SE)"] = fmt(
                    mean_lesjd, se_lesjd, is_lesjd_min
                )
                out_row["RMALA-RLMH CDLB Mean(SE)"] = fmt(mean_rl, se_rl, is_rl_min)

            rows.append(out_row)

        return pd.DataFrame(rows)

    @staticmethod
    def escape_underscores(text: str) -> str:
        """
        Escape underscores in non-LaTeX formatted strings.

        Args:
            text (str): Input string.

        Returns:
            str: String with underscores escaped.
        """
        if isinstance(text, str):
            return re.sub(r"(?<!\\)_", r"\_", text)
        return text

    @staticmethod
    def generate_latex_table(df: pd.DataFrame, output_path: str = "table.tex") -> None:
        """
        Generate a LaTeX table from a DataFrame and save it to a file.
        """
        escaped_df = df.applymap(TableGenerator.escape_underscores)

        column_format = "c" * len(escaped_df.columns)
        latex_code: str = escaped_df.to_latex(
            index=False, escape=False, column_format=column_format
        )
        with open(output_path, "w") as f:
            f.write(latex_code)
        logger.info(f"LaTeX table saved to {output_path}")

    @classmethod
    def output(
        cls,
        input: str | pd.DataFrame,
        output_path: str = "formatted_table.tex",
    ) -> None:
        """
        Read a markdown table, clean it, and output a LaTeX table.

        Args:
            input_path (str): Path to the input markdown file.
        """
        if isinstance(input, str) and input.endswith(".md"):
            # Read Markdown Table
            raw_df = cls.read_markdown_table(input)
        elif isinstance(input, pd.DataFrame):
            raw_df = input
        else:
            raise TypeError(
                "Input must be a markdown file path ending with .md or a pandas DataFrame."
            )

        # Cleaning: Remove bold markers and column name spaces
        raw_df = raw_df.replace(r"\*\*(.*?)\*\*", r"\1", regex=True)
        raw_df.columns = raw_df.columns.str.strip()

        # Replace illegal values with NaN (such as '-' or spaces)
        raw_df = raw_df.replace(r"^\s*-+\s*$", np.nan, regex=True)
        raw_df = raw_df.replace(r"^\s*$", np.nan, regex=True)

        # Convert format
        final_df = cls.apply_transformation_with_highlight(raw_df)

        cls.generate_latex_table(final_df, output_path)
