import glob
import os

import pandas as pd
from loguru import logger

RESULT_DIR = "results"


def merge_flex_mala_csv_files(
    file_pattern: str = "flex_mala_mmd_*.csv",
    output_filename: str = "merged_flex_mala_mmd.csv",
) -> pd.DataFrame | None:
    """
    Combine flex_mala_mmd_{random_seed}.csv files into a single CSV file.

    Args:
        file_pattern (str): The pattern to match the input CSV files.
        output_filename (str): The name of the output merged CSV file.

    Returns:
        pd.DataFrame | None: The merged DataFrame if successful, otherwise None.
    """
    csv_files = glob.glob(os.path.join(RESULT_DIR, file_pattern))

    if not csv_files:
        logger.warning("Did not find any files matching the pattern.")
        return None

    logger.info(f"Found the following files: {csv_files}")

    dataframes = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            logger.info(f"Success: {file}, including {len(df)} rows")
            dataframes.append(df)
        except FileNotFoundError:
            logger.warning(f"File {file} not found, skipping")
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")

    if not dataframes:
        logger.warning("Did not successfully read any files.")
        return None

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df.sort_values("random_seed", ignore_index=True)
    merged_df.to_csv(output_filename, index=False)

    logger.info("Merge complete!")
    logger.info(f"Number of rows in merged data: {len(merged_df)}")
    logger.info(f"Saved as: {output_filename}")

    # Show basic information about the merged data
    logger.info("\nPreview of merged data:")
    logger.info(merged_df.head())
    logger.info(f"\nColumns: {list(merged_df.columns)}")
    logger.info(
        f"random_seed range: {merged_df['random_seed'].min()} - {merged_df['random_seed'].max()}"
    )

    return merged_df


if __name__ == "__main__":
    result_mala = merge_flex_mala_csv_files(
        file_pattern="flex_mala_mmd_*.csv", output_filename="merged_flex_mala_mmd.csv"
    )
    print("\n" + "=" * 50 + "\n")

    result_mala_esjd = merge_flex_mala_csv_files(
        file_pattern="flex_mala_esjd_mmd_*.csv",
        output_filename="merged_flex_mala_esjd_mmd.csv",
    )
    print("\n" + "=" * 50 + "\n")
