import logging
import csv
from datetime import datetime
from ddi_graph_neural_network.config import Config
from ddi_graph_neural_network.train_model import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    feauture_list = ["__ONES__", "SMILES_GPT", "DESC_GPT", "DESC_GPT" + "_+_" + "SMILES_GPT"]
    graph_list = [
        (
            "DrugBank",
            False,
        ),
        ("CRESCENDDI", True),
        ("CRESCENDDI", False),
    ]

    datestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # create a csv file to store results
    with open(f"training_results/training_results_{datestamp}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Feature",
                "Graph",
                "Negative Samples",
                "AUC_mean",
                "PR_AUC_mean",
                "AUC_std",
                "PR_AUC_std",
                "Repetitions",
            ]
        )

    start = datetime.now()
    for graph, neg_sample in graph_list:
        for feature in feauture_list:
            print("\n================================")
            print(f"Running feature set: {feature}, Graph: {graph}, Negative Samples: {neg_sample}")

            config = Config()
            config.graph.current_graph = graph
            config.run.take_negative_samples = neg_sample
            config.training.repetitions = 5
            config.graph.feature = feature

            results = main(config)
            # write setup and results to a csv file

            with open(f"training_results/training_results_{datestamp}.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        feature,
                        graph,
                        neg_sample,
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