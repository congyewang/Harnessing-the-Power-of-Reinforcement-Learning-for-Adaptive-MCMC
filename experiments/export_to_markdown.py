from pyrlmala.utils.read import BaselineResultReader, MCMCResultReader
from pyrlmala.utils.export import PosteriorDBGenerator

posteriordb_path = "./posteriordb/posterior_database"


def generate_markdown(mode: str) -> None:
    """
    Generate a markdown file with the results of the experiments.
    """
    match mode:
        case "baseline":
            baseline_result_reader = BaselineResultReader(
                results_dir="baseline/results"
            )
            PosteriorDBGenerator(
                baseline_result_reader,
                posteriordb_path,
                output_path="baseline_results.md",
            ).execute()
        case "flex":
            mcmc_result_reader = MCMCResultReader(results_dir="whole_results")
            PosteriorDBGenerator(
                mcmc_result_reader, posteriordb_path, output_path="flex_results.md"
            ).execute()
        case _:
            raise ValueError(f"Unknown mode: {mode}. Use 'baseline' or 'flex'.")


def main() -> None:
    """
    Main function to generate markdown files for baseline and flexible MCMC results.
    """
    generate_markdown("baseline")
    generate_markdown("flex")


if __name__ == "__main__":
    main()
