from pyrlmala.utils.benchmark import BootstrapBenchmark


def main():
    model_name_list = [
        "test-laplace_1-test-laplace_1",
        "test-laplace_2-test-laplace_2",
        "test-laplace_4-test-laplace_4",
    ]
    posteriordb_path = "../posteriordb/posterior_database"

    for model_name in model_name_list:
        BootstrapBenchmark(model_name, posteriordb_path).execute()


if __name__ == "__main__":
    main()
