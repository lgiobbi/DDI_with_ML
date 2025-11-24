# %%
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.optim.lr_scheduler import MultiplicativeLR
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import structured_negative_sampling

from ddi_graph_neural_network.model import Net
from ddi_graph_neural_network.data_utils import (
    intersect_graph_and_embeddings,
    get_features_and_edges_constant,
    get_features_and_edges,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

LR_LAMBDA = 0.96
AVAILABLE_GRAPHS = {
    "ChCh_Miner": "https://raw.githubusercontent.com/liiniix/BioSNAP/master/ChCh-Miner/ChCh-Miner_durgbank-chem-chem.tsv",
    "positive_edges_CRESCENDDI": "/data/giobbi/CRESCENDDI/positive_edges_CRESCENDDI.csv",
    "CRESCENDDI": "/data/giobbi/CRESCENDDI/CRESCENDDI.csv",
}

CURRENT_GRAPH = "CRESCENDDI"

# Features
FEATURES = {
    "GPT+Desc": "/data/giobbi/embeddings/Dr_Desc_GPT.csv",
    # "SMILES_GPT": "/data/giobbi/embeddings/SMILES_GPT.csv",
    # "NoFeat (Ones)": "__ONES__",
}

DRUG_ID_NAME_MAP = {
    "GPT+Desc": "Drug ID",
    "SMILES_GPT": "DrugBank ID",
}


seed = 41
torch.manual_seed(seed)
np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch_geometric.seed_everything(seed)


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


def prepare_train_labels(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare positive and negative edge labels for training.

    Args:
        edge_index (torch.Tensor): Edge indices of the graph.
        num_nodes (int): Number of nodes in the graph.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Edge label indices and corresponding labels.
    """
    struct_neg_tup = structured_negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        contains_neg_self_loops=False,
    )
    neg_edge_index = torch.stack((struct_neg_tup[0], struct_neg_tup[2]), dim=0)
    neg_edge_index, _ = torch.unique(neg_edge_index, dim=1, return_inverse=True)

    edge_label_index = torch.cat(
        [edge_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat(
        [
            torch.ones(edge_index.size(1)),
            torch.zeros(neg_edge_index.size(1)),
        ],
        dim=0,
    )
    return edge_label_index, edge_label


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


def data_split_with_labels(data: Data) -> Tuple[Data, Data, Data]:
    """Splits the data into training, validation, and test sets while preserving labels.

    Args:
        data (Data): Input data to be split.

    Returns:
        Tuple[Data, Data, Data]: Training, validation, and test data splits.
    """
    # Todo: check if data.y does not exist if "label" column is not in the original dataframe
    if hasattr(data, "y") and data.y is not None:
        # positive samples
        positive_mask = data.y != 30
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

        return train_data, val_data, test_data
    else:
        transform = RandomLinkSplit(
            num_val=0.2,
            num_test=0.2,
            is_undirected=True,
            add_negative_train_samples=False,
            neg_sampling_ratio=1.0,
        )
        train_data, val_data, test_data = transform(data)
        edge_label_index, edge_label = prepare_train_labels(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
        )
        train_data.edge_label_index = edge_label_index
        train_data.edge_label = edge_label
        return train_data, val_data, test_data


def run_training(
    data: Data,
    device: torch.device,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 0.0003,
) -> Tuple[Net, np.ndarray, np.ndarray]:
    """Train a GNN model on the provided data.

    Args:
        data (Data): The input graph data for training.
        transform (Callable[[Data], Tuple[Data, Data, Data]]): A function to transform the data into train/val/test splits.
        device (torch.device): The device to train the model on.
        epochs (int, optional): The number of training epochs. Defaults to 100.
        patience (int, optional): The number of epochs to wait for improvement before stopping. Defaults to 10.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.0003.

    Returns:
        Tuple[Net, np.ndarray, np.ndarray]: The trained GNN model and the corresponding labels and scores.
    """
    print("-------------------------------")
    print(f"Training with LR: {lr}")

    train_data, val_data, test_data = data_split_with_labels(data)

    # get drug names of 700th edge index
    # idx = 6010
    # train_data = train_data.to(device)
    # indices_of_interest = train_data.edge_label_index[:, idx].cpu().numpy()
    # print(f"{idx} edge index drug names:", inverted_node_id_map[indices_of_interest[0]], inverted_node_id_map[indices_of_interest[1]])

    # Initialize model, optimizer, scheduler, and loss function
    model = Net(data.num_features, 256, 256).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: LR_LAMBDA)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop with early stopping
    best_val_auc: float = -float("inf")
    wait: int = 0
    best_model_state = None
    best_test_scores = None
    test_label = None
    last_test_scores = None
    last_test_label = None

    # Training loop
    for epoch in range(1, epochs):
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
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}")
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    # Fallback in case no validation improvement was recorded (e.g., tiny toy graphs)
    if best_test_scores is None:
        best_test_scores = last_test_scores if last_test_scores is not None else np.array([])
        test_label = last_test_label if last_test_label is not None else np.array([])

    return model, test_label, best_test_scores, test_data


# construct graph_data object from features and edge_index
def get_graph_data(DDI_df: pd.DataFrame, feature_type: str, feature_path: str) -> Data:
    """Construct a PyTorch Geometric Data object from DDI dataframe and features.

    Args:
        DDI_df (pd.DataFrame): DataFrame containing drug-drug interactions.
        feature_type (str): Type of features to use.
        feature_path (str): Path to the feature file or special token for constant features.

    Returns:
        Data: A PyTorch Geometric Data object containing the graph data.
    """
    DDI_df = DDI_df.copy()

    # extract features and edge_index
    if feature_path == "__ONES__":
        features, edge_index = get_features_and_edges_constant(DDI_df, feature_value=1.0, feature_dim=1)
    else:
        emb = pd.read_csv(feature_path, sep="\t", index_col=0).dropna()
        # global node_id_map
        # global inverted_node_id_map

        # def get_inverted_node_id_map(node_id_map) -> dict:
        #    inverted_map = {v: k for k, v in node_id_map.items()}
        #    return inverted_map

        DDI_df, emb, node_id_map = intersect_graph_and_embeddings(DDI_df, emb, DRUG_ID_NAME_MAP[feature_type])
        features, edge_index = get_features_and_edges(DDI_df, emb, node_id_map)
        # inverted_node_id_map = get_inverted_node_id_map(node_id_map)

    labels = torch.tensor(DDI_df["label"].values, dtype=torch.float32) if "label" in DDI_df.columns else None

    return Data(x=features, edge_index=edge_index, y=labels)


def main():
    """Train and evaluate GNN models on DDI data.

    Returns:
        dict: A dictionary containing the results of the training.
    """
    print("Current graph:", CURRENT_GRAPH)
    DDI_df = pd.read_csv(
        AVAILABLE_GRAPHS[CURRENT_GRAPH],
        sep="\t",
    ).rename(columns={"Drug1": "src", "Drug2": "dst"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    LR = [0.0003]

    results = {}
    for feature_type, feature_path in FEATURES.items():
        graph_data = get_graph_data(DDI_df, feature_type, feature_path)

        for lr in LR:
            print(f"======== {feature_type} | LR: {lr} ========")
            model, label, best_scores, test_data = run_training(graph_data, device, epochs=epochs, lr=lr)
            metrics = get_metrics(label, best_scores)

            results[(feature_type, lr)] = {
                "model": model,
                "data": graph_data,
                "metrics": metrics,
                "label": label,
                "best_scores": best_scores,
                "test_data": test_data,
            }

    for key, value in results.items():
        print(f"Model: {key[0]}, LR: {key[1]}, Metrics: {value['metrics']}")

    return results  # locals()


if __name__ == "__main__":
    main()


# %%
