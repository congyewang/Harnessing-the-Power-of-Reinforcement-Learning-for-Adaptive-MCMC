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
        Comparison is based on formatted values (scientific notation with 1 decimal place).
        If any RMALA-RLMH method has smaller value than RMALA methods, the row is flagged for highlighting.

        Args:
            df (pd.DataFrame): Input DataFrame to be transformed.

        Returns:
            pd.DataFrame: Transformed DataFrame with highlighted values and row highlight flags.
        """
        df.columns = df.columns.str.strip()
        df = df.replace(r"\*\*(.*?)\*\*", r"\1", regex=True)
        df = df.replace(r"^\s*-+\s*$", np.nan, regex=True)
        df = df.replace(r"^\s*$", np.nan, regex=True)

        def fmt(center: float, spread: float, highlight: bool = False) -> str:
            """
            Format the value and add bold if needed.
            Handles special case when formatted value is "-".
            """
            if pd.isna(center) or pd.isna(spread):
                result = "-"
            else:
                result = cls.format_scientific_with_error(center, spread)

            if result == "-":
                return "\\textbf{-}" if highlight else "-"

            return f"\\textbf{{{result}}}" if highlight else result

        def get_formatted_value(center: float, spread: float) -> float:
            """
            Parse the formatted scientific notation value as a float for comparison.
            If the value is "-", return 4.0 for comparison but preserve "-" for display.
            """
            if pd.isna(center) or pd.isna(spread):
                return 4.0

            formatted = cls.format_scientific_with_error(center, spread)

            if formatted == "-" or formatted.startswith("-"):
                return 4.0

            center_str = formatted.split("(")[0].strip()

            try:
                # Convert back to float
                return float(center_str.replace("\\times 10^{", "e").replace("}", ""))
            except ValueError:
                return 4.0

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

            med_rmala_rlmh_better = False
            mean_rmala_rlmh_better = False

            if has_median:
                try:
                    med_rl, q1_rl, q3_rl = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (
                            row["RMALA-RLMH CDLB Median"],
                            row["RMALA-RLMH CDLB Q1"],
                            row["RMALA-RLMH CDLB Q3"],
                        ),
                    )
                except (ValueError, TypeError):
                    med_rl, q1_rl, q3_rl = float("nan"), float("nan"), float("nan")

                try:
                    med_base, q1_base, q3_base = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (
                            row["RMALA AAR Median"],
                            row["RMALA AAR Q1"],
                            row["RMALA AAR Q3"],
                        ),
                    )
                except (ValueError, TypeError):
                    med_base, q1_base, q3_base = (
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )

                try:
                    med_esjd, q1_esjd, q3_esjd = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (
                            row["RMALA ESJD Median"],
                            row["RMALA ESJD Q1"],
                            row["RMALA ESJD Q3"],
                        ),
                    )
                except (ValueError, TypeError):
                    med_esjd, q1_esjd, q3_esjd = (
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )

                try:
                    med_lesjd, q1_lesjd, q3_lesjd = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (
                            row["RMALA-RLMH LESJD Median"],
                            row["RMALA-RLMH LESJD Q1"],
                            row["RMALA-RLMH LESJD Q3"],
                        ),
                    )
                except (ValueError, TypeError):
                    med_lesjd, q1_lesjd, q3_lesjd = (
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )

                formatted_med_rl = get_formatted_value(med_rl, q3_rl - q1_rl)
                formatted_med_base = get_formatted_value(med_base, q3_base - q1_base)
                formatted_med_esjd = get_formatted_value(med_esjd, q3_esjd - q1_esjd)
                formatted_med_lesjd = get_formatted_value(
                    med_lesjd, q3_lesjd - q1_lesjd
                )

                valid_med_values = []
                if not pd.isna(formatted_med_rl) and formatted_med_rl != 4.0:
                    valid_med_values.append(formatted_med_rl)
                if not pd.isna(formatted_med_base) and formatted_med_base != 4.0:
                    valid_med_values.append(formatted_med_base)
                if not pd.isna(formatted_med_esjd) and formatted_med_esjd != 4.0:
                    valid_med_values.append(formatted_med_esjd)
                if not pd.isna(formatted_med_lesjd) and formatted_med_lesjd != 4.0:
                    valid_med_values.append(formatted_med_lesjd)

                min_med_value = min(valid_med_values) if valid_med_values else None

                is_rl_min = (
                    formatted_med_rl == min_med_value
                    if min_med_value is not None
                    else False
                )
                is_lesjd_min = (
                    formatted_med_lesjd == min_med_value
                    if min_med_value is not None
                    else False
                )
                is_base_min = (
                    formatted_med_base == min_med_value
                    if min_med_value is not None
                    else False
                )
                is_esjd_min = (
                    formatted_med_esjd == min_med_value
                    if min_med_value is not None
                    else False
                )

                valid_rlmh_med = [
                    v
                    for v in [formatted_med_rl, formatted_med_lesjd]
                    if v != 4.0 and not pd.isna(v)
                ]
                valid_rmala_med = [
                    v
                    for v in [formatted_med_base, formatted_med_esjd]
                    if v != 4.0 and not pd.isna(v)
                ]

                if valid_rlmh_med and valid_rmala_med:
                    med_rmala_rlmh_better = min(valid_rlmh_med) < min(valid_rmala_med)

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
                try:
                    mean_rl, se_rl = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (row["RMALA-RLMH CDLB Mean"], row["RMALA-RLMH CDLB SE"]),
                    )
                except (ValueError, TypeError):
                    mean_rl, se_rl = float("nan"), float("nan")

                try:
                    mean_base, se_base = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (row["RMALA AAR Mean"], row["RMALA AAR SE"]),
                    )
                except (ValueError, TypeError):
                    mean_base, se_base = float("nan"), float("nan")

                try:
                    mean_esjd, se_esjd = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (row["RMALA ESJD Mean"], row["RMALA ESJD SE"]),
                    )
                except (ValueError, TypeError):
                    mean_esjd, se_esjd = float("nan"), float("nan")

                try:
                    mean_lesjd, se_lesjd = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (row["RMALA-RLMH LESJD Mean"], row["RMALA-RLMH LESJD SE"]),
                    )
                except (ValueError, TypeError):
                    mean_lesjd, se_lesjd = float("nan"), float("nan")

                formatted_mean_rl = get_formatted_value(mean_rl, se_rl)
                formatted_mean_base = get_formatted_value(mean_base, se_base)
                formatted_mean_esjd = get_formatted_value(mean_esjd, se_esjd)
                formatted_mean_lesjd = get_formatted_value(mean_lesjd, se_lesjd)

                valid_mean_values = []
                if not pd.isna(formatted_mean_rl) and formatted_mean_rl != 4.0:
                    valid_mean_values.append(formatted_mean_rl)
                if not pd.isna(formatted_mean_base) and formatted_mean_base != 4.0:
                    valid_mean_values.append(formatted_mean_base)
                if not pd.isna(formatted_mean_esjd) and formatted_mean_esjd != 4.0:
                    valid_mean_values.append(formatted_mean_esjd)
                if not pd.isna(formatted_mean_lesjd) and formatted_mean_lesjd != 4.0:
                    valid_mean_values.append(formatted_mean_lesjd)

                min_mean_value = min(valid_mean_values) if valid_mean_values else None

                is_rl_min = (
                    formatted_mean_rl == min_mean_value
                    if min_mean_value is not None
                    else False
                )
                is_lesjd_min = (
                    formatted_mean_lesjd == min_mean_value
                    if min_mean_value is not None
                    else False
                )
                is_base_min = (
                    formatted_mean_base == min_mean_value
                    if min_mean_value is not None
                    else False
                )
                is_esjd_min = (
                    formatted_mean_esjd == min_mean_value
                    if min_mean_value is not None
                    else False
                )

                valid_rlmh_mean = [
                    v
                    for v in [formatted_mean_rl, formatted_mean_lesjd]
                    if v != 4.0 and not pd.isna(v)
                ]
                valid_rmala_mean = [
                    v
                    for v in [formatted_mean_base, formatted_mean_esjd]
                    if v != 4.0 and not pd.isna(v)
                ]

                if valid_rlmh_mean and valid_rmala_mean:
                    mean_rmala_rlmh_better = min(valid_rlmh_mean) < min(
                        valid_rmala_mean
                    )

                out_row["RMALA AAR Mean(SE)"] = fmt(mean_base, se_base, is_base_min)
                out_row["RMALA ESJD Mean(SE)"] = fmt(mean_esjd, se_esjd, is_esjd_min)
                out_row["RMALA-RLMH LESJD Mean(SE)"] = fmt(
                    mean_lesjd, se_lesjd, is_lesjd_min
                )
                out_row["RMALA-RLMH CDLB Mean(SE)"] = fmt(mean_rl, se_rl, is_rl_min)

            row_needs_gray = med_rmala_rlmh_better or mean_rmala_rlmh_better

            out_row["_needs_gray"] = row_needs_gray

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
        Adds row coloring for rows that need highlighting.
        """
        gray_rows = df["_needs_gray"] if "_needs_gray" in df.columns else []

        if "_needs_gray" in df.columns:
            df = df.drop("_needs_gray", axis=1)

        escaped_df = df.applymap(TableGenerator.escape_underscores)

        column_format = "c" * len(escaped_df.columns)

        header = escaped_df.columns.tolist()
        rows = escaped_df.values.tolist()

        latex_lines = [
            "\\begin{tabular}{" + column_format + "}",
            "\\hline",
            " & ".join(header) + " \\\\",
            "\\hline",
        ]

        for i, row in enumerate(rows):
            if i < len(gray_rows) and gray_rows.iloc[i]:
                latex_lines.append("\\rowcolor{gray!20}")

            latex_lines.append(" & ".join(str(cell) for cell in row) + " \\\\")

        latex_lines.extend(["\\hline", "\\end{tabular}"])

        latex_code = "\n".join(latex_lines)

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


class SimplifiedTableGenerator:
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
    def apply_simple_transformation(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformation to the DataFrame to format values in scientific notation
        and highlight (bold) the smaller value when comparing RLBarker LESJD and RLBarker CDLB.

        Args:
            df (pd.DataFrame): Input DataFrame to be transformed.

        Returns:
            pd.DataFrame: Transformed DataFrame with formatted values.
        """
        df.columns = df.columns.str.strip()
        df = df.replace(r"\*\*(.*?)\*\*", r"\1", regex=True)
        df = df.replace(r"^\s*-+\s*$", np.nan, regex=True)
        df = df.replace(r"^\s*$", np.nan, regex=True)

        def fmt(center: float, spread: float, highlight: bool = False) -> str:
            """
            Format the value in scientific notation and add bold if highlighted.
            """
            if pd.isna(center) or pd.isna(spread):
                result = "-"
            else:
                result = cls.format_scientific_with_error(center, spread)

            return f"\\textbf{{{result}}}" if highlight else result

        def get_formatted_value(center: float, spread: float) -> float:
            """
            Parse the formatted scientific notation value as a float for comparison.
            If the value is "-", return float('inf') for comparison.
            """
            if pd.isna(center) or pd.isna(spread):
                return float("inf")

            if center == 0:
                return 0.0

            return abs(center)  # Using absolute value for comparison

        has_median = all(
            col in df.columns
            for col in [
                "RLBarker LESJD Median",
                "RLBarker CDLB Median",
            ]
        )
        has_mean = all(
            col in df.columns
            for col in [
                "RLBarker LESJD Mean",
                "RLBarker CDLB Mean",
            ]
        )

        rows: List[Dict[str, str]] = []

        for _, row in df.iterrows():
            model: str = row["Model"]
            dim = int(row["d"]) if "d" in row and not pd.isna(row["d"]) else None
            out_row = {"Model": model, "d": dim}

            if has_median:
                try:
                    med_rl, q1_rl, q3_rl = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (
                            row["RLBarker CDLB Median"],
                            row["RLBarker CDLB Q1"],
                            row["RLBarker CDLB Q3"],
                        ),
                    )
                except (ValueError, TypeError):
                    med_rl, q1_rl, q3_rl = float("nan"), float("nan"), float("nan")

                try:
                    med_lesjd, q1_lesjd, q3_lesjd = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (
                            row["RLBarker LESJD Median"],
                            row["RLBarker LESJD Q1"],
                            row["RLBarker LESJD Q3"],
                        ),
                    )
                except (ValueError, TypeError):
                    med_lesjd, q1_lesjd, q3_lesjd = (
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )

                # Compare median values for highlighting
                lesjd_med_value = get_formatted_value(med_lesjd, q3_lesjd - q1_lesjd)
                cdlb_med_value = get_formatted_value(med_rl, q3_rl - q1_rl)

                # Determine which one is smaller for highlighting
                is_lesjd_smaller = lesjd_med_value < cdlb_med_value
                is_cdlb_smaller = cdlb_med_value < lesjd_med_value

                # If both are equal (or both are NaN), neither gets highlighted

                out_row["RLBarker LESJD Mid(IQR)"] = fmt(
                    med_lesjd, q3_lesjd - q1_lesjd, is_lesjd_smaller
                )
                out_row["RLBarker CDLB Mid(IQR)"] = fmt(
                    med_rl, q3_rl - q1_rl, is_cdlb_smaller
                )

            if has_mean:
                try:
                    mean_rl, se_rl = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (row["RLBarker CDLB Mean"], row["RLBarker CDLB SE"]),
                    )
                except (ValueError, TypeError):
                    mean_rl, se_rl = float("nan"), float("nan")

                try:
                    mean_lesjd, se_lesjd = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (row["RLBarker LESJD Mean"], row["RLBarker LESJD SE"]),
                    )
                except (ValueError, TypeError):
                    mean_lesjd, se_lesjd = float("nan"), float("nan")

                # Compare mean values for highlighting
                lesjd_mean_value = get_formatted_value(mean_lesjd, se_lesjd)
                cdlb_mean_value = get_formatted_value(mean_rl, se_rl)

                # Determine which one is smaller for highlighting
                is_lesjd_smaller = lesjd_mean_value < cdlb_mean_value
                is_cdlb_smaller = cdlb_mean_value < lesjd_mean_value

                # If both are equal (or both are NaN), neither gets highlighted
                out_row["RLBarker LESJD Mean(SE)"] = fmt(
                    mean_lesjd, se_lesjd, is_lesjd_smaller
                )
                out_row["RLBarker CDLB Mean(SE)"] = fmt(mean_rl, se_rl, is_cdlb_smaller)

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
        No highlighting or bold text.
        """
        escaped_df = df.applymap(SimplifiedTableGenerator.escape_underscores)
        column_format = "c" * len(escaped_df.columns)
        header = escaped_df.columns.tolist()
        rows = escaped_df.values.tolist()

        latex_lines = [
            "\\begin{tabular}{" + column_format + "}",
            "\\hline",
            " & ".join(header) + " \\\\",
            "\\hline",
        ]

        for row in rows:
            latex_lines.append(" & ".join(str(cell) for cell in row) + " \\\\")

        latex_lines.extend(["\\hline", "\\end{tabular}"])

        latex_code = "\n".join(latex_lines)

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
        Read a markdown table, clean it, format values, and output a LaTeX table.

        Args:
            input (str|pd.DataFrame): Path to the input markdown file or a pandas DataFrame.
            output_path (str): Path where the output LaTeX table will be saved.
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
        final_df = cls.apply_simple_transformation(raw_df)

        cls.generate_latex_table(final_df, output_path)


class EnhancedTableGenerator:
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

    @staticmethod
    def get_formatted_value(center: float, spread: float) -> float:
        """
        Parse the formatted scientific notation value as a float for comparison.
        If the value is "-", return float('inf') for comparison.
        """
        if pd.isna(center) or pd.isna(spread):
            return float("inf")

        if center == 0:
            return 0.0

        return abs(center)  # Using absolute value for comparison

    @classmethod
    def apply_enhanced_transformation(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformation to the DataFrame to format values in scientific notation
        and highlight (bold) the smallest value when comparing Barker, RLBarker LESJD, and RLBarker CDLB.

        Also tracks if RMALA-RLMH methods have smaller metrics than both RMALA methods.

        Args:
            df (pd.DataFrame): Input DataFrame to be transformed.

        Returns:
            pd.DataFrame: Transformed DataFrame with formatted values and row highlighting info.
        """
        df.columns = df.columns.str.strip()
        df = df.replace(r"\*\*(.*?)\*\*", r"\1", regex=True)
        df = df.replace(r"^\s*-+\s*$", np.nan, regex=True)
        df = df.replace(r"^\s*$", np.nan, regex=True)

        def fmt(center: float, spread: float, highlight: bool = False) -> str:
            """
            Format the value in scientific notation and add bold if highlighted.
            """
            if pd.isna(center) or pd.isna(spread):
                result = "-"
            else:
                result = cls.format_scientific_with_error(center, spread)

            return f"\\textbf{{{result}}}" if highlight else result

        has_median = all(
            col in df.columns
            for col in [
                "RLBarker LESJD Median",
                "RLBarker CDLB Median",
            ]
        )

        has_barker_median = "Barker Median" in df.columns

        has_mean = all(
            col in df.columns
            for col in [
                "RLBarker LESJD Mean",
                "RLBarker CDLB Mean",
            ]
        )

        has_barker_mean = "Barker Mean" in df.columns

        # Check for RMALA-RLMH methods
        rmala_rlmh_columns = [col for col in df.columns if "RMALA-RLMH" in col]
        has_rmala_rlmh = len(rmala_rlmh_columns) > 0

        rows: List[Dict[str, str]] = []
        row_highlights: List[bool] = []

        for _, row in df.iterrows():
            model: str = row["Model"]
            dim = int(row["d"]) if "d" in row and not pd.isna(row["d"]) else None
            out_row = {"Model": model, "d": dim}

            # Initialize all output metrics to None
            barker_mid_iqr = None
            lesjd_mid_iqr = None
            cdlb_mid_iqr = None
            barker_mean_se = None
            lesjd_mean_se = None
            cdlb_mean_se = None

            # Default value for row highlighting
            should_highlight_row = False

            if has_median:
                # Process RLBarker CDLB metrics
                try:
                    med_rl, q1_rl, q3_rl = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (
                            row["RLBarker CDLB Median"],
                            row["RLBarker CDLB Q1"],
                            row["RLBarker CDLB Q3"],
                        ),
                    )
                except (ValueError, TypeError):
                    med_rl, q1_rl, q3_rl = float("nan"), float("nan"), float("nan")

                # Process RLBarker LESJD metrics
                try:
                    med_lesjd, q1_lesjd, q3_lesjd = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (
                            row["RLBarker LESJD Median"],
                            row["RLBarker LESJD Q1"],
                            row["RLBarker LESJD Q3"],
                        ),
                    )
                except (ValueError, TypeError):
                    med_lesjd, q1_lesjd, q3_lesjd = (
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )

                # Process Barker metrics if available
                med_barker, q1_barker, q3_barker = (
                    float("nan"),
                    float("nan"),
                    float("nan"),
                )
                if has_barker_median:
                    try:
                        med_barker, q1_barker, q3_barker = map(
                            lambda x: float(x) if not pd.isna(x) else float("nan"),
                            (
                                row["Barker Median"],
                                row["Barker Q1"],
                                row["Barker Q3"],
                            ),
                        )
                    except (ValueError, TypeError):
                        med_barker, q1_barker, q3_barker = (
                            float("nan"),
                            float("nan"),
                            float("nan"),
                        )

                # Get formatted values for comparison
                lesjd_med_value = cls.get_formatted_value(
                    med_lesjd, q3_lesjd - q1_lesjd
                )
                cdlb_med_value = cls.get_formatted_value(med_rl, q3_rl - q1_rl)
                barker_med_value = (
                    cls.get_formatted_value(med_barker, q3_barker - q1_barker)
                    if has_barker_median
                    else float("inf")
                )

                # Compare all three values to find the smallest
                values = [
                    (lesjd_med_value, "RLBarker LESJD"),
                    (cdlb_med_value, "RLBarker CDLB"),
                ]

                if has_barker_median:
                    values.append((barker_med_value, "Barker"))

                min_value, min_method = min(values, key=lambda x: x[0])

                # Check if all values are equal
                all_equal = all(
                    val[0] == min_value for val in values if not np.isinf(val[0])
                )

                # Format metrics with bold for the smallest value (unless all are equal)
                if has_barker_median:
                    barker_mid_iqr = fmt(
                        med_barker,
                        q3_barker - q1_barker,
                        (min_method == "Barker" and not all_equal),
                    )

                lesjd_mid_iqr = fmt(
                    med_lesjd,
                    q3_lesjd - q1_lesjd,
                    (min_method == "RLBarker LESJD" and not all_equal),
                )

                cdlb_mid_iqr = fmt(
                    med_rl,
                    q3_rl - q1_rl,
                    (min_method == "RLBarker CDLB" and not all_equal),
                )

                # Check RMALA-RLMH metrics for row highlighting
                if has_rmala_rlmh:
                    # Find RMALA-RLMH median columns
                    rmala_rlmh_med_cols = [
                        col for col in rmala_rlmh_columns if "Median" in col
                    ]
                    for col in rmala_rlmh_med_cols:
                        try:
                            med_rmala_rlmh = float(row[col])
                            # Find corresponding Q1 and Q3 columns
                            base_col = col.replace(" Median", "")
                            q1_col = f"{base_col} Q1"
                            q3_col = f"{base_col} Q3"
                            if q1_col in df.columns and q3_col in df.columns:
                                q1_rmala_rlmh = float(row[q1_col])
                                q3_rmala_rlmh = float(row[q3_col])

                                rmala_rlmh_value = cls.get_formatted_value(
                                    med_rmala_rlmh, q3_rmala_rlmh - q1_rmala_rlmh
                                )

                                # If RMALA-RLMH has smaller metric than both RMALA methods
                                if (
                                    rmala_rlmh_value < lesjd_med_value
                                    and rmala_rlmh_value < cdlb_med_value
                                ):
                                    should_highlight_row = True
                                    break
                        except (ValueError, TypeError):
                            continue

            if has_mean:
                # Process RLBarker CDLB metrics
                try:
                    mean_rl, se_rl = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (row["RLBarker CDLB Mean"], row["RLBarker CDLB SE"]),
                    )
                except (ValueError, TypeError):
                    mean_rl, se_rl = float("nan"), float("nan")

                # Process RLBarker LESJD metrics
                try:
                    mean_lesjd, se_lesjd = map(
                        lambda x: float(x) if not pd.isna(x) else float("nan"),
                        (row["RLBarker LESJD Mean"], row["RLBarker LESJD SE"]),
                    )
                except (ValueError, TypeError):
                    mean_lesjd, se_lesjd = float("nan"), float("nan")

                # Process Barker metrics if available
                mean_barker, se_barker = float("nan"), float("nan")
                if has_barker_mean:
                    try:
                        mean_barker, se_barker = map(
                            lambda x: float(x) if not pd.isna(x) else float("nan"),
                            (row["Barker Mean"], row["Barker SE"]),
                        )
                    except (ValueError, TypeError):
                        mean_barker, se_barker = float("nan"), float("nan")

                # Get formatted values for comparison
                lesjd_mean_value = cls.get_formatted_value(mean_lesjd, se_lesjd)
                cdlb_mean_value = cls.get_formatted_value(mean_rl, se_rl)
                barker_mean_value = (
                    cls.get_formatted_value(mean_barker, se_barker)
                    if has_barker_mean
                    else float("inf")
                )

                # Compare all three values to find the smallest
                values = [
                    (lesjd_mean_value, "RLBarker LESJD"),
                    (cdlb_mean_value, "RLBarker CDLB"),
                ]

                if has_barker_mean:
                    values.append((barker_mean_value, "Barker"))

                min_value, min_method = min(values, key=lambda x: x[0])

                # Check if all values are equal
                all_equal = all(
                    val[0] == min_value for val in values if not np.isinf(val[0])
                )

                # Format metrics with bold for the smallest value (unless all are equal)
                if has_barker_mean:
                    barker_mean_se = fmt(
                        mean_barker,
                        se_barker,
                        (min_method == "Barker" and not all_equal),
                    )

                lesjd_mean_se = fmt(
                    mean_lesjd,
                    se_lesjd,
                    (min_method == "RLBarker LESJD" and not all_equal),
                )

                cdlb_mean_se = fmt(
                    mean_rl, se_rl, (min_method == "RLBarker CDLB" and not all_equal)
                )

                # Check RMALA-RLMH metrics for row highlighting
                if has_rmala_rlmh and not should_highlight_row:
                    # Find RMALA-RLMH mean columns
                    rmala_rlmh_mean_cols = [
                        col for col in rmala_rlmh_columns if "Mean" in col
                    ]
                    for col in rmala_rlmh_mean_cols:
                        try:
                            mean_rmala_rlmh = float(row[col])
                            # Find corresponding SE column
                            base_col = col.replace(" Mean", "")
                            se_col = f"{base_col} SE"
                            if se_col in df.columns:
                                se_rmala_rlmh = float(row[se_col])

                                rmala_rlmh_value = cls.get_formatted_value(
                                    mean_rmala_rlmh, se_rmala_rlmh
                                )

                                # If RMALA-RLMH has smaller metric than both RMALA methods
                                if (
                                    rmala_rlmh_value < lesjd_mean_value
                                    and rmala_rlmh_value < cdlb_mean_value
                                ):
                                    should_highlight_row = True
                                    break
                        except (ValueError, TypeError):
                            continue

            # Add columns in the desired order
            # First, add the required columns Model and d
            ordered_output = {"Model": model, "d": dim}

            # Add Barker columns if available
            if has_barker_median:
                ordered_output["Barker Mid(IQR)"] = barker_mid_iqr
            if has_barker_mean:
                ordered_output["Barker Mean(SE)"] = barker_mean_se

            # Add RLBarker LESJD columns
            if has_median:
                ordered_output["RLBarker LESJD Mid(IQR)"] = lesjd_mid_iqr
            if has_mean:
                ordered_output["RLBarker LESJD Mean(SE)"] = lesjd_mean_se

            # Add RLBarker CDLB columns
            if has_median:
                ordered_output["RLBarker CDLB Mid(IQR)"] = cdlb_mid_iqr
            if has_mean:
                ordered_output["RLBarker CDLB Mean(SE)"] = cdlb_mean_se

            rows.append(ordered_output)
            row_highlights.append(should_highlight_row)

        result_df = pd.DataFrame(rows)
        result_df.attrs["row_highlights"] = row_highlights
        return result_df

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
        Applies row coloring based on the row_highlights attribute.
        """
        escaped_df = df.applymap(EnhancedTableGenerator.escape_underscores)
        column_format = "c" * len(escaped_df.columns)
        header = escaped_df.columns.tolist()
        rows = escaped_df.values.tolist()

        # Get row highlighting information
        row_highlights = df.attrs.get("row_highlights", [False] * len(rows))

        latex_lines = [
            "\\begin{tabular}{" + column_format + "}",
            "\\hline",
            " & ".join(header) + " \\\\",
            "\\hline",
        ]

        for i, row in enumerate(rows):
            # Add row coloring command if needed
            if i < len(row_highlights) and row_highlights[i]:
                latex_lines.append("\\rowcolor{gray!20}")

            latex_lines.append(" & ".join(str(cell) for cell in row) + " \\\\")

        latex_lines.extend(["\\hline", "\\end{tabular}"])

        latex_code = "\n".join(latex_lines)

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
        Read a markdown table, clean it, format values, and output a LaTeX table.

        Args:
            input (str|pd.DataFrame): Path to the input markdown file or a pandas DataFrame.
            output_path (str): Path where the output LaTeX table will be saved.
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

        # Apply enhanced transformation
        final_df = cls.apply_enhanced_transformation(raw_df)

        # Generate LaTeX table with row highlighting
        cls.generate_latex_table(final_df, output_path)
