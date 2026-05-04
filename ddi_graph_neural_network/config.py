import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

DATA_DIR = Path(os.getenv("DDI_DATA_DIR", "/data/giobbi"))
BASE_PATH_FEATURES = f"{DATA_DIR}/embeddings/"
AVAILABLE_FEATURES: List[str] = ["DESC_GPT", "SMILES_GPT", "DESC_LLAMAII7b"]

class LossType(Enum):
    """Types of loss functions available."""

    BCEWithLogitsLoss = "BCEWithLogitsLoss"
    FocalLoss = "FocalLoss"
    WeightedBCEWithLogitsLoss = "WeightedBCEWithLogitsLoss"


@dataclass
class RunSettings:
    """Current run settings."""

    take_negative_samples: bool = False  # If graph has negative samples, whether to include them
    balanced_labels: bool = (
        False  # If graph has negative samples, whether to balance pos/neg samples to min(num_pos, num_neg)
    )
    upsample_negative_labels: bool = False  # Whether to upsample negative samples to balance labels
    use_only_sampled_negatives_in_train: bool = (
        False  # Whether to drop negative samples during training (only use for evaluation)
    )

    loss_type: LossType = LossType.BCEWithLogitsLoss

    # WeightedBCEWithLogitsLoss and FocalLoss specific parameters

    # Multiplier for positive class in WeightedBCEWithLogitsLoss.
    # Loss of positive samples *= pos_loss_multiplier * (num_negatives / num_positives)
    pos_loss_multiplier: float = 0.5

    # FocalLoss specific parameters
    # focal_loss_alpha: float = 0.25  # Alpha > 1 increases (< 1 decreases) the importance of positive samples
    focal_loss_gamma: float = (
        2.0  # Reduce the relative loss for well-classified examples. 0.0 is equivalent to WeightedBCEWithLogitsLoss
    )


@dataclass
class GraphParams:
    """Graph-related configuration."""

    current_graph: str = "CRESCENDDI"
    feature: str = "DESC_GPT"
    col_name_drug_id: str = "Drug ID"

    seed_graph_sampling: int = 32

    available_graphs: Dict[str, str] = field(
        default_factory=lambda: {
            "DrugBank": f"{DATA_DIR}/GRAPH/drugbank_graph.csv",  # "https://raw.githubusercontent.com/liiniix/BioSNAP/master/ChCh-Miner/ChCh-Miner_durgbank-chem-chem.tsv",
            "positive_edges_CRESCENDDI": f"{DATA_DIR}/CRESCENDDI/positive_edges_CRESCENDDI.csv",
            "CRESCENDDI": f"{DATA_DIR}/CRESCENDDI/CRESCENDDI_wo_contradiction.csv",
            "DrugBank_CRESCENDDI": f"{DATA_DIR}/GRAPH/drugbank_crescenddi_graph_wo_contradiction.csv",
            "ogbl-ddi": str(DATA_DIR),
        }
    )

    @property
    def available_features(self) -> List[str]:
        combinations = [
            f1 + "_+_" + f2 for i, f1 in enumerate(AVAILABLE_FEATURES) for f2 in AVAILABLE_FEATURES[i:] if f1 != f2
        ]
        return AVAILABLE_FEATURES + ["__ONES__"] + combinations

    @property
    def feature_path(self) -> str:
        if self.feature not in self.available_features:
            raise ValueError(f"Feature '{self.feature}' is not among available features: {self.available_features}")
        return BASE_PATH_FEATURES + self.feature + ".csv"

@dataclass
class TrainingParams:
    """Training-related configuration."""

    learning_rate: float = 0.0003
    epochs: int = 100
    patience: int = 10
    lr_lambda: float = 0.96
    repetitions: int = 1
    seed: Optional[int] = None


@dataclass
class Config:
    """Top-level configuration object composed of grouped params.

    Usage examples:
        cfg = Config()
        cfg.graph.current_graph = "CRESCENDDI"
        cfg.training.learning_rate = 1e-3
    """

    graph: GraphParams = field(default_factory=GraphParams)
    training: TrainingParams = field(default_factory=TrainingParams)
    run: RunSettings = field(default_factory=RunSettings)