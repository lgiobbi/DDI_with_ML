# %%
import warnings
from typing import Tuple
import logging


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.optim.lr_scheduler import MultiplicativeLR
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import structured_negative_sampling
from torchvision.ops import sigmoid_focal_loss
from ddi_graph_neural_network.model import Net
from ddi_graph_neural_network.data_utils import get_graph_data


from ddi_graph_neural_network.config import Config, LossType

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = sigmoid_focal_loss(logits, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        return loss


def train(
    model: Net,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_data: Data,
) -> torch.Tensor:
    """
    Trains the GNN model for one epoch on the provided training data.

    Args:
        model (Net): The GNN model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.Module): Loss function to optimize.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_data (Data): Training data containing node features and edges.

    Returns:
        torch.Tensor: The computed loss value for the epoch.
    """
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    predicted_edge_label = model.decode(z, train_data.edge_label_index).view(-1)
    loss = criterion(predicted_edge_label, train_data.edge_label)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss


@torch.no_grad()
def test(model: Net, data: Data) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the GNN model on the test data.

    Args:
        model (Net): The GNN model to be evaluated.
        data (Data): The test data containing node features and edges.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: The ROC AUC score, true labels, and predicted scores.
    """
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    roc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    label = data.edge_label.cpu().numpy()
    score = out.cpu().numpy()
    return roc, label, score


def get_metrics(label: np.ndarray, scores: np.ndarray) -> dict:
    """Calculate evaluation metrics based on true labels and predicted scores.

    Args:
        label (np.ndarray): True labels.
        scores (np.ndarray): Predicted scores.

    Returns:
        dict: A dictionary containing AUC and PR_AUC metrics.
    """
    auc_score = roc_auc_score(label, scores)
    precision, recall, _ = precision_recall_curve(label, scores)
    pr_auc = auc(recall, precision)
    return {"AUC": auc_score, "PR_AUC": pr_auc}


def combine_splits(pos_split, neg_split):
    """Combines positive and negative data splits."""
    combined_split = pos_split.clone()

    # The supervision edges from the negative split are all negatives (label 0)
    neg_split.edge_label = torch.zeros_like(neg_split.edge_label)

    # Concatenate supervision edges and labels
    combined_split.edge_label_index = torch.cat([pos_split.edge_label_index, neg_split.edge_label_index], dim=-1)
    combined_split.edge_label = torch.cat([pos_split.edge_label, neg_split.edge_label], dim=0)

    return combined_split


def sample_negative_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Samples negative edges using structured negative sampling."""
    struct_neg_tup = structured_negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        contains_neg_self_loops=False,
    )
    neg_edge_index = torch.stack((struct_neg_tup[0], struct_neg_tup[2]), dim=0)
    neg_edge_index, _ = torch.unique(neg_edge_index, dim=1, return_inverse=True)
    return neg_edge_index

def _split_without_real_negatives(config: Config, data: Data) -> Tuple[Data, Data, Data]:
    """Splits the data into training, validation, and test sets by sampling all negative edges."""
    logger.debug(f"No negative edges in the graph data. Sampling {data.edge_index.shape[1]} negative edges.")
    transform = RandomLinkSplit(
        num_val=0.2,
        num_test=0.2,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = transform(data)
    # Sample negative edges for training data
    neg_edge_index = sample_negative_edges(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
    )

    edge_label_index = torch.cat([train_data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat(
        [
            torch.ones(train_data.edge_index.size(1)),
            torch.zeros(neg_edge_index.size(1)),
        ],
        dim=0,
    )

    train_data.edge_label_index = edge_label_index
    train_data.edge_label = edge_label

    return train_data, val_data, test_data


def _further_process_train_with_real_negatives(config: Config, train_data: Data) -> Data:
    """Further processes the training data when real negative samples are present, based on the configuration settings."""
    if config.run.use_only_sampled_negatives_in_train:
        # Drop negative samples from training data (only use for evaluation)
        train_data.edge_label_index = train_data.edge_label_index[:, train_data.edge_label == 1]
        train_data.edge_label = train_data.edge_label[train_data.edge_label == 1]
        logger.debug(
            f"Dropped negative samples from training data. "
            f"Training data now has {train_data.edge_label.sum().item()} positive samples and {(train_data.edge_label == 0).sum().item()} negative samples."
        )

    if config.run.upsample_negative_labels or config.run.use_only_sampled_negatives_in_train:
        # Add negative samples to balance labels in training data
        num_pos = (train_data.edge_label == 1).sum().item()
        num_neg = (train_data.edge_label == 0).sum().item()
        if num_pos < num_neg:
            logger.debug("Not possible to upsample negative samples when positives are fewer.")
        else:
            neg_edge_index = sample_negative_edges(
                edge_index=train_data.edge_index,
                num_nodes=train_data.num_nodes,
            )

            num_neg_to_sample = num_pos - num_neg
            logger.debug(f"Upsampling negative samples by {num_neg_to_sample} to balance labels.")
            perm = torch.randperm(neg_edge_index.size(1))[:num_neg_to_sample]
            neg_edge_index = neg_edge_index[:, perm]

            edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
            edge_label = torch.cat(
                [
                    train_data.edge_label,
                    torch.zeros(neg_edge_index.size(1)),
                ],
                dim=0,
            )
            train_data.edge_label_index = edge_label_index
            train_data.edge_label = edge_label
            logger.debug(
                f"After upsampling, training data has {int(train_data.edge_label.sum().item())} positive and {(train_data.edge_label == 0).sum().item()} negative samples."
            )
    return train_data


def _split_with_real_negatives(config: Config, data: Data) -> Tuple[Data, Data, Data]:
    """Splits the data into training, validation, and test sets using real negative labels."""
    # positive samples
    positive_mask = data.y == 1
    positive_data = Data(x=data.x, edge_index=data.edge_index[:, positive_mask])  # , y=data.y[positive_mask])
    positive_transform = RandomLinkSplit(
        num_val=0.2,
        num_test=0.2,
        is_undirected=True,  # adds bidirectional edges if not already present
        add_negative_train_samples=False,  # should the train set have negative samples added like val/test?
        neg_sampling_ratio=0.0,  # No new negative samples
        # key="y",
    )
    pos_train, pos_val, pos_test = positive_transform(positive_data)

    # negative samples
    negative_mask = data.y == 0
    negative_data = Data(x=data.x, edge_index=data.edge_index[:, negative_mask])  # , y=data.y[negative_mask])
    negative_transform = RandomLinkSplit(
        num_val=0.2,
        num_test=0.2,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=0.0,
        # key="y",
    )
    neg_train, neg_val, neg_test = negative_transform(negative_data)

    # Create the final combined data splits
    train_data = combine_splits(pos_train, neg_train)
    val_data = combine_splits(pos_val, neg_val)
    test_data = combine_splits(pos_test, neg_test)

    if config.run.upsample_negative_labels or config.run.use_only_sampled_negatives_in_train:
        train_data = _further_process_train_with_real_negatives(config, train_data)

    return train_data, val_data, test_data


def data_split_with_labels(config: Config, data: Data) -> Tuple[Data, Data, Data]:
    """Splits the data into training, validation, and test sets while preserving labels.

    Args:
        config (Config): Configuration object containing settings.
        data (Data): Input data to be split.

    Returns:
        Tuple[Data, Data, Data]: Training, validation, and test data splits.
    """
    if config.graph.seed_graph_sampling is not None:
        # If seed is set for graph sampling but not for training
        cpu_state = torch.get_rng_state()
        torch.manual_seed(config.graph.seed_graph_sampling)

    try:
        # Todo: check if data.y does not exist if "label" column is not in the original dataframe
        if hasattr(data, "y") and data.y is not None:
            return _split_with_real_negatives(config, data)
        else:
            return _split_without_real_negatives(config, data)
    finally:
        if config.graph.seed_graph_sampling is not None:
            torch.set_rng_state(cpu_state)

def get_criterion(config: Config, train_data: Data, device: torch.device) -> torch.nn.Module:
    """Get the loss criterion based on the configuration."""
    match config.run.loss_type:
        case LossType.BCEWithLogitsLoss:
            return torch.nn.BCEWithLogitsLoss()
        case LossType.WeightedBCEWithLogitsLoss:
            num_positives = (train_data.edge_label == 1).sum().item()
            num_negatives = (train_data.edge_label == 0).sum().item()
            imbalance_neg_over_pos = torch.tensor([num_negatives / (num_positives)], device=device)
            pos_weight = imbalance_neg_over_pos * config.run.pos_loss_multiplier

            logger.debug(f"Using imbalanced loss with pos_weight: {(pos_weight.item()):.4f}")
            return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        case LossType.FocalLoss:
            num_positives = (train_data.edge_label == 1).sum().item()
            num_negatives = (train_data.edge_label == 0).sum().item()
            imbalance_neg_over_pos = torch.tensor([num_negatives / (num_positives)], device=device)
            pos_weight = imbalance_neg_over_pos * config.run.pos_loss_multiplier

            logger.debug(
                f"Using imbalanced focal loss with pos_weight: {(pos_weight.item()):.4f}, gamma: {config.run.focal_loss_gamma}"
            )
            return SigmoidFocalLoss(alpha=pos_weight, gamma=config.run.focal_loss_gamma)
        case _:
            raise ValueError(f"Unsupported loss type: {config.run.loss_type}")

def run_training(
    config: Config,
    data: Data,
    device: torch.device,
) -> Tuple[Net, np.ndarray, np.ndarray]:
    """Train a GNN model on the provided data.

    Args:
        config (Config): Configuration object containing training parameters.
        data (Data): The input graph data for training.
        device (torch.device): The device to run the training on.

    Returns:
        Tuple[Net, np.ndarray, np.ndarray]: The trained GNN model and the corresponding labels and scores.
    """

    train_data, val_data, test_data = data_split_with_labels(config, data)

    # get drug names of 700th edge index
    # idx = 6010
    # train_data = train_data.to(device)
    # indices_of_interest = train_data.edge_label_index[:, idx].cpu().numpy()
    # print(f"{idx} edge index drug names:", inverted_node_id_map[indices_of_interest[0]], inverted_node_id_map[indices_of_interest[1]])

    # Initialize model, optimizer, scheduler, and loss function
    model = Net(data.num_features, 256, 256).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.learning_rate)
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: config.training.lr_lambda)
    criterion = get_criterion(config, train_data, device)

    # Training loop with early stopping
    best_val_auc: float = -float("inf")
    wait: int = 0
    best_model_state = None
    best_test_scores = None
    test_label = None
    last_test_scores = None
    last_test_label = None

    # Training loop
    for epoch in range(1, config.training.epochs):
        loss = train(
            model,
            optimizer,
            criterion,
            scheduler,
            train_data,
        )
        val_auc, _, _ = test(model, val_data)
        _, test_label_tmp, test_score_tmp = test(model, test_data)
        last_test_scores = test_score_tmp
        last_test_label = test_label_tmp
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_scores = test_score_tmp
            test_label = test_label_tmp
            wait = 0
            best_model_state = model.state_dict()
        else:
            wait += 1
        # print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}")
        if wait >= config.training.patience:
            logger.debug(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    # Fallback in case no validation improvement was recorded (e.g., tiny toy graphs)
    if best_test_scores is None:
        best_test_scores = last_test_scores if last_test_scores is not None else np.array([])
        test_label = last_test_label if last_test_label is not None else np.array([])

    return model, test_label, best_test_scores, test_data


def main(config: Config = Config()) -> dict:
    """Train and evaluate GNN models on DDI data.

    Returns:
        dict: A dictionary containing the results of the training.
    """
    if config.training.seed is not None:
        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)
        torch_geometric.seed_everything(config.training.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    print("Current graph:", config.graph.current_graph)
    DDI_df = pd.read_csv(
        config.graph.available_graphs[config.graph.current_graph],
        sep="\t",
    ).rename(columns={"Drug1": "src", "Drug2": "dst"})

    # shuffle the dataframe
    DDI_df = DDI_df.sample(frac=1, random_state=config.graph.seed_graph_sampling).reset_index(
        drop=True
    )  # , random_state=10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    AUC_bucket = []
    PR_bucket = []
    graph_data, node_id_map = get_graph_data(DDI_df, config)

    print(f"======== {config.graph.feature} ========")
    for i in range(config.training.repetitions):
        logger.debug(
            f"Run {i + 1}/{config.training.repetitions} for {config.graph.feature} | LR: {config.training.learning_rate}"
        )
        model, label, test_scores, test_data = run_training(
            config,
            graph_data,
            device,
        )
        metrics = get_metrics(label, test_scores)
        AUC_bucket.append(metrics["AUC"])
        PR_bucket.append(metrics["PR_AUC"])

    print("-------------------------------")
    print(f"-- FINAL RESULTS FOR GRAPH {config.graph.current_graph} | FEATURE {config.graph.feature} -- ")
    print("Graph Data: ", config.graph.current_graph)
    print(f"ROC_AUC: {np.mean(AUC_bucket):.4f}")
    print(f"PR_AUC: {np.mean(PR_bucket):.4f}")

    if config.training.repetitions > 1:
        print(f"std ROC_AUC: {np.std(AUC_bucket):.4f}")
        print(f"std PR_AUC: {np.std(PR_bucket):.4f}")
        print(f"repetitions: {config.training.repetitions}")
        print("-------------------------------")

    metrics = {
        "AUC_mean": np.mean(AUC_bucket),
        "AUC_std": np.std(AUC_bucket),
        "PR_AUC_mean": np.mean(PR_bucket),
        "PR_AUC_std": np.std(PR_bucket),
        "repetitions": config.training.repetitions,
    }

    results = {
        "config": config,
        "model": model,
        "data": graph_data,
        "metrics": metrics,
        "label": label,
        "test_scores": test_scores,
        "test_data": test_data,
        "node_id_map": node_id_map,
    }
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    config = Config()
    config.run.take_negative_samples = True
    config.run.balanced_labels = False
    config.run.upsample_negative_labels = False
    config.run.use_only_sampled_negatives_in_train = True
    config.run.loss_type = LossType.WeightedBCEWithLogitsLoss
    config.run.pos_loss_multiplier = 1

    config.training.seed = 42
    config.graph.seed_graph_sampling = 42
    config.graph.current_graph = "DrugBank_CRESCENDDI"
    config.graph.feature = "DESC_LLAMAII7b"

    run = main(config)

# %%
