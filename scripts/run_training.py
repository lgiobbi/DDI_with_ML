import logging
import csv
from copy import deepcopy
from datetime import datetime
from ddi_graph_neural_network.config import Config, RunSettings
from ddi_graph_neural_network.train_model import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    config_list = []

    config = Config()
    config.graph.current_graph = "DrugBank_CRESCENDDI"
    config.training.repetitions = 5

    run_settings = [
        RunSettings(take_negative_samples=False, imbalanced_loss=False),
        RunSettings(take_negative_samples=True, imbalanced_loss=True),
        RunSettings(take_negative_samples=True, imbalanced_loss=False),
    ]
    for run in run_settings:
        config.run = run
        config_list.append(deepcopy(config))

    datestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # create a csv file to store results
    with open(f"training_results/training_results_{datestamp}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                # "Feature",
                # "Graph",
                "Take Negative Samples",
                "Take Imbalanced Loss",
                "AUC_mean",
                "PR_AUC_mean",
                "AUC_std",
                "PR_AUC_std",
                "Repetitions",
            ]
        )

    start = datetime.now()

    for config in config_list:
        print("\n================================")
        print(f"Running with settings: {config.run}")

        results = main(config)

        with open(f"training_results/training_results_{datestamp}.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    config.run.take_negative_samples,
                    config.run.imbalanced_loss,
                    results["metrics"]["AUC_mean"],
                    results["metrics"]["PR_AUC_mean"],
                    results["metrics"]["AUC_std"],
                    results["metrics"]["PR_AUC_std"],
                    results["metrics"]["repetitions"],
                ]
            )

        print("================================\n")
    end = datetime.now()
    print("Total runtime:", end - start)