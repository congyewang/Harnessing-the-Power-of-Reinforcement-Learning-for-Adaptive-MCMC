import os

import pandas as pd

from pyrlmala.utils import Toolbox
from pyrlmala.utils.export import PosteriorDBGenerator, TableGenerator
from pyrlmala.utils.read import BaselineResultReader, MCMCResultReader


def get_dataframe() -> pd.DataFrame:
    """
    Generate a DataFrame containing the results from different methods.

    Returns:
        pd.DataFrame: A DataFrame containing the results from different methods.
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

    baseline_aar_df = baseline_aar_exporter.get_result_dataframe()
    baseline_esjd_df = baseline_esjd_exporter.get_result_dataframe()
    rl_cdlb_df = rl_cdlb_exporter.get_result_dataframe()
    rl_esjd_df = rl_esjd_exporter.get_result_dataframe()

    merged_df = pd.merge(baseline_aar_df, baseline_esjd_df, on=["Model", "d"])
    merged_df = pd.merge(merged_df, rl_cdlb_df, on=["Model", "d"])
    merged_df = pd.merge(merged_df, rl_esjd_df, on=["Model", "d"])

    return merged_df


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
    df = get_dataframe()

    match mode:
        case "mean":
            mean_df = get_mean_sub_dataframe(df)
            mean_output_file_path = os.path.join(output_root_dir, "mean_results.tex")

            Toolbox.create_folder(mean_output_file_path)

            TableGenerator.output(mean_df, mean_output_file_path)
        case "median":
            median_df = get_median_sub_dataframe(df)
            median_output_file_path = os.path.join(
                output_root_dir, "median_results.tex"
            )

            Toolbox.create_folder(median_output_file_path)

            TableGenerator.output(median_df, median_output_file_path)
        case "all":
            median_df = get_median_sub_dataframe(df)
            mean_df = get_mean_sub_dataframe(df)

            mean_output_file_path = os.path.join(output_root_dir, "mean_results.tex")
            median_output_file_path = os.path.join(
                output_root_dir, "median_results.tex"
            )

            Toolbox.create_folder(mean_output_file_path)
            Toolbox.create_folder(median_output_file_path)

            TableGenerator.output(mean_df, mean_output_file_path)
            TableGenerator.output(median_df, median_output_file_path)
        case _:
            raise ValueError(f"Unknown mode: {mode}. Use 'mean', 'median' or 'all'.")


def main() -> None:
    """
    Main function to generate LaTeX tables for the results of the experiments.
    """
    output_tex("mean", ".")


if __name__ == "__main__":
    main()
