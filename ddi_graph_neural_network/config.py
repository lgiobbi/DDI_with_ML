BASE_PATH_FEATURES = "/data/giobbi/embeddings/"
AVAILABLE_FEATURES = ["DESC_GPT", "SMILES_GPT"]


class Config:
    def __init__(
        self,
        current_graph="CRESCENDDI",
        feature="DESC_GPT",
        take_negative_samples=False,
        repetitions=1,
        epochs=100,
        patience=10,
        learning_rate=0.0003,
        lr_lambda=0.96,
        seed=None,
        seed_graph_sampling=32,
    ):
        self.feature = feature
        self.repetitions = repetitions

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.lr_lambda = lr_lambda

        # current settings
        self.current_graph = current_graph
        self.take_negative_samples = take_negative_samples
        self.seed = seed
        self.seed_graph_sampling = seed_graph_sampling

        # column names
        self.col_name_drug_id = "Drug ID"

        self.available_graphs = {
            "DrugBank": "/data/giobbi/GRAPH/drugbank_graph.csv",  # "https://raw.githubusercontent.com/liiniix/BioSNAP/master/ChCh-Miner/ChCh-Miner_durgbank-chem-chem.tsv",
            "positive_edges_CRESCENDDI": "/data/giobbi/CRESCENDDI/positive_edges_CRESCENDDI.csv",
            "CRESCENDDI": "/data/giobbi/CRESCENDDI/CRESCENDDI_wo_contradiction.csv",  # Exclude DB positives from negatives     "/data/giobbi/CRESCENDDI/CRESCENDDI.csv",
        }

    @property
    def _available_features(self):
        combinations = [
            f1 + "_+_" + f2 for i, f1 in enumerate(AVAILABLE_FEATURES) for f2 in AVAILABLE_FEATURES[i:] if f1 != f2
        ]
        return AVAILABLE_FEATURES + ["__ONES__"] + combinations

    @property
    def feature_path(self):
        if self.feature not in self._available_features:
            raise ValueError(f"Feature '{self.feature}' is not among available features: {self._available_features}")
        return BASE_PATH_FEATURES + self.feature + ".csv"
