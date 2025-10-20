# %%
from typing import Callable, List, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.optim.lr_scheduler import MultiplicativeLR
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import structured_negative_sampling

warnings.simplefilter(action="ignore", category=FutureWarning)

LR_LAMBDA = 0.96


def PyG_data(feature: np.ndarray, DDI_graph: pd.DataFrame, node_id_map: dict) -> Data:
    """Prepare the PyTorch Geometric data object.

    Args:
        feature (np.ndarray): Node features.
        DDI_graph (pd.DataFrame): Drug-drug interaction graph.
        node_id_map (dict): Mapping from drug IDs to node indices.

    Returns:
        Data: PyTorch Geometric data object.
    """

    DDI_graph = DDI_graph.map(lambda id: map_node_id(node_id_map, id)).to_numpy()
    DDI_graph = np.vstack((DDI_graph, DDI_graph[:, ::-1]))  # Make bidirectional

    data = Data(
        x=torch.tensor(feature, dtype=torch.float32),
        edge_index=torch.tensor(DDI_graph).t().contiguous(),
    )
    return data


def get_node_id_map(DDI_graph: pd.DataFrame) -> dict:
    """Get a mapping from drug IDs to node indices.

    Args:
        DDI_graph (pd.DataFrame): Drug-drug interaction graph.

    Returns:
        dict: Mapping from drug IDs to node indices.
    """
    DrugIDs_in_graph = np.unique(DDI_graph.values)
    node_id_map = {node_name: i for i, node_name in enumerate(DrugIDs_in_graph)}
    return node_id_map


def map_node_id(node_id_map: dict, drug_id: str) -> int | None:
    """Map a drug ID to its corresponding node index.

    Args:
        node_id_map (dict): Mapping from drug IDs to node indices.
        drug_id (str): Drug ID to map.

    Returns:
        int | None: Corresponding node index or None if not found.
    """
    return node_id_map.get(drug_id, None)


class Net(torch.nn.Module):
    """Graph Neural Network model for link prediction."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        """Initialize the GNN model.

        Args:
            in_channels (int): Number of input features.
            hidden_channels (int): Number of hidden features.
            out_channels (int): Number of output features.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode the input features using GCN layers.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Graph connectivity.

        Returns:
            torch.Tensor: Encoded node features.
        """
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.3)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            z (torch.Tensor): _description_
            edge_label_index (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z: torch.Tensor) -> torch.Tensor:
        """Decode all node pairs.

        Args:
            z (torch.Tensor): Encoded node features.

        Returns:
            torch.Tensor: Decoded edge indices.
        """
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the GNN.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity.
            edge_label_index (torch.Tensor): Edge labels.

        Returns:
            torch.Tensor: Predicted edge labels.
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


def train(
    model: Net,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_data: Data,
    edge_label_index: torch.Tensor,
    edge_label_gt: torch.Tensor,
) -> torch.Tensor:
    """
    Trains the GNN model for one epoch on the provided training data.

    Args:
        model (Net): The GNN model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.Module): Loss function to optimize.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        train_data (Data): Training data containing node features and edges.
        edge_label_index (torch.Tensor): Edge indices for which to compute the loss.
        edge_label_gt (torch.Tensor): Ground truth labels for the edges.

    Returns:
        torch.Tensor: The computed loss value for the epoch.
    """
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    predicted_edge_label = model.decode(z, edge_label_index).view(-1)
    loss = criterion(predicted_edge_label, edge_label_gt)
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


def no_feature(
    smiles: List[str], DDI_graph: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Generate features for the given SMILES strings and DDI graph.

    Args:
        smiles (List[str]): List of SMILES strings.
        DDI_graph (pd.DataFrame): DataFrame representing the DDI graph.

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: A tuple containing the generated features and the original DDI graph.
    """
    features = np.ones((len(smiles), 100))
    print("no_feature")
    return features, DDI_graph


# Train a single model and return model, train/val/test data, and metrics
def run_training(
    data: Data,
    transform: Callable[[Data], Tuple[Data, Data, Data]],
    device: torch.device,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 0.0003,
) -> Tuple[Net, dict]:
    """Train a GNN model on the provided data.

    Args:
        data (Data): The input data for training.
        transform (Callable[[Data], Tuple[Data, Data, Data]]): A function to transform the data into train/val/test splits.
        device (torch.device): The device to train the model on.
        epochs (int, optional): The number of training epochs. Defaults to 100.
        patience (int, optional): The number of epochs to wait for improvement before stopping. Defaults to 10.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.0003.

    Returns:
        Tuple[Net, dict]: The trained GNN model and a dictionary of metrics.
    """
    print("-------------------------------")
    print(f"Training with LR: {lr}")
    train_data, val_data, test_data = transform(data)
    model = Net(data.num_features, 256, 256).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: LR_LAMBDA)
    criterion = torch.nn.BCEWithLogitsLoss()

    struct_neg_tup = structured_negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        contains_neg_self_loops=False,
    )
    neg_edge_index = torch.stack((struct_neg_tup[0], struct_neg_tup[2]), dim=0)
    neg_edge_index, _ = torch.unique(neg_edge_index, dim=1, return_inverse=True)

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat(
        [
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1)),
        ],
        dim=0,
    )

    best_val_auc = final_test_auc = 0
    wait = 0
    best_model_state = None
    for epoch in range(1, epochs):
        loss = train(
            model,
            optimizer,
            criterion,
            scheduler,
            train_data,
            edge_label_index,
            edge_label,
        )
        val_auc, _, _ = test(model, val_data)
        test_auc, label, score = test(model, test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
            best_scores = score
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

    precision, recall, _ = precision_recall_curve(label, best_scores)
    pr = auc(recall, precision)
    metrics = {"AUC": final_test_auc, "PR_AUC": pr}
    return model, metrics


def main():
    """Train and evaluate GNN models on DDI data.

    Returns:
        dict: A dictionary containing the results of the training.
    """
    models = {"GPT+Desc": "/data/giobbi/embeddings/Dr_Desc_GPT.csv"}

    # Data loading
    DDI_graph = pd.read_csv(
        "https://raw.githubusercontent.com/liiniix/BioSNAP/master/ChCh-Miner/ChCh-Miner_durgbank-chem-chem.tsv",
        sep="\t",
    ).rename(columns={"Drug1": "src", "Drug2": "dst"})

    transform = RandomLinkSplit(
        num_val=0.2,
        num_test=0.2,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    LR = [0.0003]

    results = {}
    node_id_map = get_node_id_map(DDI_graph)
    for modelname, dir in models.items():
        emb = pd.read_csv(dir, sep="\t", index_col=0)
        emb = emb.select_dtypes(include=["float"])

        graph_with_emb = PyG_data(emb.values, DDI_graph, node_id_map)

        for lr in LR:
            print(f"======== {modelname} | LR: {lr} ========")

            model, metrics = run_training(
                graph_with_emb, transform, device, epochs=epochs, lr=lr
            )
            results[(modelname, lr)] = {
                "model": model,
                "data": graph_with_emb,
                "metrics": metrics,
            }

    for key, value in results.items():
        print(f"Model: {key[0]}, LR: {key[1]}, Metrics: {value['metrics']}")

    return locals()


if __name__ == "__main__":
    main()


# %%
