import os
from typing import Tuple

import pandas as pd

from pyrlmala.utils import Toolbox
from pyrlmala.utils.export import (
    EnhancedTableGenerator,
    PosteriorDBGenerator,
    SingleTableGenerator,
    TableGenerator,
)
from pyrlmala.utils.read import BaselineResultReader, MCMCResultReader


def get_dataframe() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate DataFrames for RMALA and RLBarker results.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing three DataFrames:
            - RMALA results DataFrame
            - RLBarker results DataFrame
            - Sensitivity analysis results DataFrame
    """
    baseline_aar_reader = BaselineResultReader(
        results_dir="./baseline", method_name="RMALA AAR"
    )
    baseline_esjd_reader = BaselineResultReader(
        results_dir="./baseline_esjd", method_name="RMALA ESJD"
    )
    rl_cdlb_reader = MCMCResultReader(
        results_dir="./whole_results_cdlb", method_name="RMALA-RLMH CDLB"
    )
    rl_esjd_reader = MCMCResultReader(
        results_dir="./whole_results_esjd", method_name="RMALA-RLMH LESJD"
    )

    baseline_barker_reader = BaselineResultReader(
        results_dir="./baseline_barker", method_name="Barker"
    )
    rl_barker_cdlb_reader = MCMCResultReader(
        results_dir="./whole_results_barker_cdlb", method_name="RLBarker CDLB"
    )
    rl_barker_esjd_reader = MCMCResultReader(
        results_dir="./whole_results_barker_esjd", method_name="RLBarker LESJD"
    )
    rl_sensitive_reader = MCMCResultReader(
        results_dir="./sensitivity", method_name="RMALA-RLMH CDLB"
    )

    baseline_aar_exporter = PosteriorDBGenerator(
        baseline_aar_reader, "./posteriordb/posterior_database"
    )
    baseline_esjd_exporter = PosteriorDBGenerator(
        baseline_esjd_reader, "./posteriordb/posterior_database"
    )
    rl_cdlb_exporter = PosteriorDBGenerator(
        rl_cdlb_reader, "./posteriordb/posterior_database"
    )
    rl_esjd_exporter = PosteriorDBGenerator(
        rl_esjd_reader, "./posteriordb/posterior_database"
    )

    baseline_barker_exporter = PosteriorDBGenerator(
        baseline_barker_reader, "./posteriordb/posterior_database"
    )
    rl_barker_cdlb_exporter = PosteriorDBGenerator(
        rl_barker_cdlb_reader, "./posteriordb/posterior_database"
    )
    rl_barker_esjd_exporter = PosteriorDBGenerator(
        rl_barker_esjd_reader, "./posteriordb/posterior_database"
    )
    rl_sensitive_exporter = PosteriorDBGenerator(
        rl_sensitive_reader, "./posteriordb/posterior_database"
    )

    baseline_aar_df = baseline_aar_exporter.get_result_dataframe()
    baseline_esjd_df = baseline_esjd_exporter.get_result_dataframe()
    rl_cdlb_df = rl_cdlb_exporter.get_result_dataframe()
    rl_esjd_df = rl_esjd_exporter.get_result_dataframe()

    baseline_barker_df = baseline_barker_exporter.get_result_dataframe()
    rl_barker_cdlb_df = rl_barker_cdlb_exporter.get_result_dataframe()
    rl_barker_esjd_df = rl_barker_esjd_exporter.get_result_dataframe()
    rl_sensitive_cdlb_df = rl_sensitive_exporter.get_result_dataframe()

    merged_df_mala = pd.merge(baseline_aar_df, baseline_esjd_df, on=["Model", "d"])
    merged_df_mala = pd.merge(merged_df_mala, rl_cdlb_df, on=["Model", "d"])
    merged_df_mala = pd.merge(merged_df_mala, rl_esjd_df, on=["Model", "d"])

    merged_df_barker = pd.merge(
        baseline_barker_df, rl_barker_cdlb_df, on=["Model", "d"]
    )
    merged_df_barker = pd.merge(merged_df_barker, rl_barker_esjd_df, on=["Model", "d"])

    return merged_df_mala, merged_df_barker, rl_sensitive_cdlb_df


def get_mean_sub_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame to include only columns with 'Mean' or 'SE' in their names.

    Args:
        df (pd.DataFrame): The DataFrame to filter.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the columns with 'Mean' or 'SE'.
    """
    return df.filter(regex="Model|^d$|Mean|SE")


def get_median_sub_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame to include only columns with 'Median', 'Q1', or 'Q3' in their names.

    Args:
        df (pd.DataFrame): The DataFrame to filter.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the columns with 'Median', 'Q1', or 'Q3'.
    """
    return df.filter(regex="Model|^d$|Median|Q1|Q3")


def output_tex(mode: str = "mean", output_root_dir: str = ".") -> None:
    """
    Generate a LaTeX table from the DataFrame and save it to a file.

    Args:
        mode (str): The mode to use for generating the table. Can be 'mean', 'median', or 'all'.
        output_root_dir (str): The root directory where the output file will be saved.
    """
    df_mala, df_barker, df_sensitive = get_dataframe()

    match mode:
        case "mean":
            mean_df_mala = get_mean_sub_dataframe(df_mala)
            mean_output_file_path_mala = os.path.join(
                output_root_dir, "mean_results.tex"
            )
            Toolbox.create_folder(mean_output_file_path_mala)
            TableGenerator.output(mean_df_mala, mean_output_file_path_mala)

            mean_df_barker = get_mean_sub_dataframe(df_barker)
            mean_output_file_path_barker = os.path.join(
                output_root_dir, "mean_results_barker.tex"
            )
            Toolbox.create_folder(mean_output_file_path_barker)
            EnhancedTableGenerator.output(mean_df_barker, mean_output_file_path_barker)

            mean_df_sensitive = get_mean_sub_dataframe(df_sensitive)
            mean_output_file_path_sensitive = os.path.join(
                output_root_dir, "mean_results_sensitive.tex"
            )
            Toolbox.create_folder(mean_output_file_path_sensitive)
            SingleTableGenerator.output(
                mean_df_sensitive, mean_output_file_path_sensitive
            )
        case "median":
            median_df_mala = get_median_sub_dataframe(df_mala)
            median_output_file_path_mala = os.path.join(
                output_root_dir, "median_results.tex"
            )
            Toolbox.create_folder(median_output_file_path_mala)
            TableGenerator.output(median_df_mala, median_output_file_path_mala)

            median_df_barker = get_median_sub_dataframe(df_barker)
            median_output_file_path_barker = os.path.join(
                output_root_dir, "median_results_barker.tex"
            )
            Toolbox.create_folder(median_output_file_path_barker)
            EnhancedTableGenerator.output(
                median_df_barker, median_output_file_path_barker
            )

            median_df_sensitive = get_median_sub_dataframe(df_sensitive)
            median_output_file_path_sensitive = os.path.join(
                output_root_dir, "median_results_sensitive.tex"
            )
            Toolbox.create_folder(median_output_file_path_sensitive)
            SingleTableGenerator.output(
                median_df_sensitive, median_output_file_path_sensitive
            )
        case "all":
            median_df_mala = get_median_sub_dataframe(df_mala)
            mean_df_mala = get_mean_sub_dataframe(df_mala)

            mean_output_file_path_mala = os.path.join(
                output_root_dir, "mean_results.tex"
            )
            median_output_file_path_mala = os.path.join(
                output_root_dir, "median_results.tex"
            )

            Toolbox.create_folder(mean_output_file_path_mala)
            Toolbox.create_folder(median_output_file_path_mala)

            TableGenerator.output(mean_df, mean_output_file_path_mala)
            TableGenerator.output(median_df, median_output_file_path_mala)

            median_df_barker = get_median_sub_dataframe(df_barker)
            mean_df_barker = get_mean_sub_dataframe(df_barker)

            mean_output_file_path_barker = os.path.join(
                output_root_dir, "mean_results_barker.tex"
            )
            median_output_file_path_barker = os.path.join(
                output_root_dir, "median_results_barker.tex"
            )

            Toolbox.create_folder(mean_output_file_path_barker)
            Toolbox.create_folder(median_output_file_path_barker)

            EnhancedTableGenerator.output(mean_df_barker, mean_output_file_path_barker)
            EnhancedTableGenerator.output(
                median_df_barker, median_output_file_path_barker
            )

            median_df_sensitive = get_median_sub_dataframe(df_sensitive)
            mean_df_sensitive = get_mean_sub_dataframe(df_sensitive)
            mean_output_file_path_sensitive = os.path.join(
                output_root_dir, "mean_results_sensitive.tex"
            )
            median_output_file_path_sensitive = os.path.join(
                output_root_dir, "median_results_sensitive.tex"
            )
            Toolbox.create_folder(mean_output_file_path_sensitive)
            Toolbox.create_folder(median_output_file_path_sensitive)
            SingleTableGenerator.output(
                mean_df_sensitive, mean_output_file_path_sensitive
            )
            SingleTableGenerator.output(
                median_df_sensitive, median_output_file_path_sensitive
            )
        case _:
            raise ValueError(f"Unknown mode: {mode}. Use 'mean', 'median' or 'all'.")


def main() -> None:
    """
    Main function to generate LaTeX tables for the results of the experiments.
    """
    output_tex("all", ".")


if __name__ == "__main__":
    main()
