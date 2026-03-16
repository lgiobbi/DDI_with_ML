from typing import Tuple
import numpy as np
import pandas as pd
import torch
import logging

from torch_geometric.utils import coalesce
from torch_geometric.data import Data
from ddi_graph_neural_network.config import Config

KEPT_PERC_NOT_IN_GRAPH = 0.0  # Percentage of drugs not in graph to keep

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_node_id_map(DDI_graph: pd.DataFrame) -> dict:
    """Get a mapping from drug IDs to node indices.

    Args:
        DDI_graph (pd.DataFrame): Drug-drug interaction graph.

    Returns:
        dict: Mapping from drug IDs to node indices.
    """
    DrugIDs_in_graph = np.unique(DDI_graph[["src", "dst"]].values)
    node_id_map = {node_name: i for i, node_name in enumerate(DrugIDs_in_graph)}
    return node_id_map


def _map_node_id(node_id_map: dict, drug_id: str) -> int | None:
    """Map a drug ID to its corresponding node index.

    Args:
        node_id_map (dict): Mapping from drug IDs to node indices.
        drug_id (str): Drug ID to map.

    Returns:
        int | None: Corresponding node index or None if not found.
    """
    return node_id_map.get(drug_id, None)


def _match_embeddings_to_graph(
    DDI_graph: pd.DataFrame, emb: pd.DataFrame, drug_id_col: str
) -> Tuple[pd.DataFrame, dict]:
    """Align embeddings to the DDI graph, adding dummy embeddings for missing drugs.

    Args:
        DDI_graph (pd.DataFrame):  The drug-drug interaction graph.
        emb (pd.DataFrame): The drug embeddings.
        drug_id_col (str): The column name for drug IDs in the embeddings.
    Returns:
        Tuple[pd.DataFrame, dict]: The aligned embeddings and node ID mapping.
    """
    all_drug_ids = pd.unique(DDI_graph[["src", "dst"]].values.ravel())
    emb = emb.set_index(drug_id_col)
    missing_drugs = set(all_drug_ids) - set(emb.index)
    if len(missing_drugs) > 0:
        dummy = pd.DataFrame(
            1,
            index=list(missing_drugs),
            columns=emb.columns,
        )
        emb = pd.concat([emb, dummy])
    node_id_map = {drug_id: i for i, drug_id in enumerate(all_drug_ids)}
    emb = emb.reindex(all_drug_ids)
    return emb, node_id_map


def _intersect_graph_and_embeddings(
    DDI_graph: pd.DataFrame, emb: pd.DataFrame, drug_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Drop drug from the graph that are not in the embeddings and drop embeddings that are not in the graph.
    Align embeddings to the order of drugs in the graph.

    Args:
        DDI_graph (pd.DataFrame): The drug-drug interaction graph.
        emb (pd.DataFrame): The drug embeddings.
        drug_id_col (str): The column name for drug IDs in the embeddings.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, dict]: The aligned DDI graph, embeddings, and node ID mapping.
    """
    number_of_graph_drugs_not_in_emb = len(set(DDI_graph["src"]).union(set(DDI_graph["dst"])) - set(emb[drug_id_col])) 

    DDI_graph = DDI_graph[
        DDI_graph["src"].isin(emb[drug_id_col]) & DDI_graph["dst"].isin(emb[drug_id_col])
    ].reset_index(drop=True)

    DrugIDs_in_graph = np.unique(DDI_graph[["src", "dst"]].values)

    # emb = emb[emb['Drug ID'].isin(DrugIDs_in_graph)]

    # Create a mapping from drug IDs to node indices
    node_id_map = {drug_id: i for i, drug_id in enumerate(DrugIDs_in_graph)}

    # Align embeddings to node_id_map order
    emb = emb.set_index(drug_id_col)
    # emb = emb.reindex(DrugIDs_in_graph)

    in_graph = emb.loc[emb.index.intersection(DrugIDs_in_graph)]
    not_in_graph = emb.loc[~emb.index.isin(DrugIDs_in_graph)]

    kept_n_not_in_graph = int(len(not_in_graph) * KEPT_PERC_NOT_IN_GRAPH / 100.0)

    logger.debug(
        "Graph and Embeddings Intersection Summary:\n"
        f"Number of drugs in graph: {len(DrugIDs_in_graph)}, \n"
        f"Number of embedding drugs in graph: {len(in_graph)}, \n"
        f"Number of drugs dropped from graph (not in embeddings): {number_of_graph_drugs_not_in_emb}, \n"
        f"Number of embedding drugs not in graph: {len(not_in_graph)}\n"
        f"Kept percentage of embedding drugs not in graph: {KEPT_PERC_NOT_IN_GRAPH}%, \n"
    )    

    not_in_graph = not_in_graph.iloc[:kept_n_not_in_graph, :]

    # Stack: first those in graph (in order), then the rest
    emb = pd.concat([in_graph.reindex(DrugIDs_in_graph).dropna(how="all"), not_in_graph])

    if len(DrugIDs_in_graph) != len(emb):
        print(
            "\n---------------------------\n"
            " Warning: Mismatch in number of drugs between graph and embeddings!\n"
            "---------------------------\n"
        )

    return DDI_graph, emb, node_id_map


def _get_features_and_edges(
    DDI_df: pd.DataFrame,
    emb: pd.DataFrame,
    node_id_map: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process the DDI graph and embeddings to prepare edge indices.

    Args:
        DDI_graph (pd.DataFrame): The drug-drug interaction graph.
        emb (pd.DataFrame): The drug embeddings.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The processed edge index and node features.
    """
    emb = emb.select_dtypes(include=["float"])
    features = torch.tensor(emb.values, dtype=torch.float32)

    edge_index = DDI_df[["src", "dst"]].map(lambda id: _map_node_id(node_id_map, id)).to_numpy()
    edge_index = torch.tensor(edge_index).t().contiguous()

    # DDI_graph = np.vstack((DDI_graph, DDI_graph[:, ::-1]))  # Make bidirectional
    return features, edge_index


def _get_features_and_edges_constant(
    DDI_df: pd.DataFrame,
    feature_value: float = 1.0,
    feature_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build features and edge index using only the graph structure.

    Nodes receive a constant feature vector (default: a single 1), enabling a
    "featureless" GCN run that learns purely from topology.

    Args:
        DDI_graph (pd.DataFrame): The drug-drug interaction graph with columns [src, dst].
        feature_value (float, optional): The constant value for each feature. Defaults to 1.0.
        feature_dim (int, optional): Number of repeated constant features. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Node features tensor and edge_index tensor.
    """
    node_id_map = _get_node_id_map(DDI_df)
    num_nodes = len(node_id_map)
    features = torch.full((num_nodes, feature_dim), float(feature_value), dtype=torch.float32)

    edge_index = DDI_df[["src", "dst"]].map(lambda id: _map_node_id(node_id_map, id)).to_numpy()
    edge_index = torch.tensor(edge_index).t().contiguous()

    return features, edge_index

# construct graph_data object from features and edge_index
def _get_graph_data_from_csv(DDI_df: pd.DataFrame, config: Config) -> Tuple[Data, dict]:
    """Construct a PyTorch Geometric Data object from DDI dataframe and features.

    Args:
        DDI_df (pd.DataFrame): DataFrame containing drug-drug interactions.
        config (Config): Configuration object.

    Returns:
        Tuple[Data, dict]: A PyTorch Geometric Data object and node_id_map.
    """
    feature_type = config.graph.feature
    DDI_df = DDI_df.copy()

    # extract features and edge_index
    if feature_type == "__ONES__":
        node_id_map = _get_node_id_map(DDI_df)
        features, edge_index = _get_features_and_edges_constant(DDI_df, feature_value=1.0, feature_dim=1)
    else:
        emb = pd.read_csv(config.graph.feature_path, sep="\t", index_col=0).dropna()
        DDI_df, emb, node_id_map = _intersect_graph_and_embeddings(DDI_df, emb, config.graph.col_name_drug_id)
        features, edge_index = _get_features_and_edges(DDI_df, emb, node_id_map)
        # inverted_node_id_map = get_inverted_node_id_map(node_id_map)

    labels = torch.tensor(DDI_df["label"].values, dtype=torch.float32) if "label" in DDI_df.columns else None

    # If labels are present, optionally balance positives/negatives and/or drop explicit negatives
    if labels is not None:
        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        if config.run.balanced_labels:
            min_count = min(pos_mask.sum().item(), neg_mask.sum().item())
            pos_indices = pos_indices[:min_count]
            neg_indices = neg_indices[:min_count]
            logger.debug(
                f"Balancing the dataset to have equal positive and negative samples. \n"
                f"Dropped positive edges: {pos_mask.sum().item() - min_count}, \n"
                f"Dropped negative edges: {neg_mask.sum().item() - min_count}, \n"
            )

        if config.run.take_negative_samples:
            selected_indices = torch.cat([pos_indices, neg_indices])
            labels = labels[selected_indices]
        else:
            logger.debug(f"Taking only positive samples. \nDropped negative edges: {neg_indices.numel()}, \n")
            selected_indices = pos_indices
            labels = None

        edge_index = edge_index[:, selected_indices]

    # Ensure canonical edge order (sorted by source, then dest) to make results invariant to input shuffle
    if labels is not None:
        edge_index, labels = coalesce(edge_index, labels, len(node_id_map))
    else:
        edge_index = coalesce(edge_index, num_nodes=len(node_id_map))

    logger.debug(f"Final graph has {features.size(0)} nodes and {edge_index.size(1)} edges.\n"
                f"Positive edges: {int(labels.sum().item()) if labels is not None else edge_index.size(1)}, \n"
                f"Negative edges: {int((labels == 0).sum().item()) if labels is not None else 0}"
                )

    return Data(x=features, edge_index=edge_index, y=labels), node_id_map


def _get_graph_data_from_ogbl_ddi(
    config: Config,
) -> Tuple[Data, dict]:
    """Load ogbl-ddi dataset and create a Data object with predefined train/val/test splits.

    Args:
        config (Config): Configuration object.

    Returns:
        Tuple[Data, dict]: A Data object with split information stored as attributes, and node_id_map.
    """
    from ogb.linkproppred import PygLinkPropPredDataset

    dataset_root = config.graph.available_graphs.get(config.graph.current_graph)
    feature_type = config.graph.feature

    # Patch torch.load BEFORE importing OGB to handle weights_only default
    # This is required because PyTorch 2.6 defaults weights_only=True, which breaks OGB dataset loading
    _torch_load_orig = torch.load.__wrapped__ if hasattr(torch.load, "__wrapped__") else torch.load

    def torch_load_patched(f, *args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _torch_load_orig(f, *args, **kwargs)

    torch.load = torch_load_patched

    # Load dataset
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=dataset_root)
    ogbl_data = dataset[0]

    # Get predefined splits
    split_edges = dataset.get_edge_split()
    train_edge_index = split_edges["train"]["edge"].t().contiguous()
    val_edge_index = split_edges["valid"]["edge"].t().contiguous()
    test_edge_index = split_edges["test"]["edge"].t().contiguous()

    # Load node ID mapping from ogbl-ddi mapping file
    mapping_path = f"{dataset_root}/ogbl_ddi/mapping/nodeidx2drugid.csv.gz"
    node_df = pd.read_csv(mapping_path, compression="gzip")
    node_id_map = {row["drug id"]: row["node idx"] for _, row in node_df.iterrows()}

    # Create node features (constant ones if not specified)
    if feature_type == "__ONES__":
        features = torch.ones((ogbl_data.num_nodes, 1), dtype=torch.float32)
    else:
        # Load and align real embeddings
        emb = pd.read_csv(config.graph.feature_path, sep="\t", index_col=0).dropna()
        emb = emb.select_dtypes(include=["float"])

        # Create a mapping dictionary: node_idx -> drug_id
        idx_to_drug = {idx: drug_id for drug_id, idx in node_id_map.items()}
        ordered_drug_ids = [idx_to_drug[i] for i in range(ogbl_data.num_nodes)]
        emb = emb.reindex(ordered_drug_ids)
        emb = emb.fillna(1.0)

        features = torch.tensor(emb.values, dtype=torch.float32)

    # Combine all edges with split labels
    # Label: 2=train, 1=valid, 0=test (for compatibility with existing logic)

    all_edges = torch.cat([train_edge_index, val_edge_index, test_edge_index], dim=1)
    split_labels = torch.cat(
        [
            torch.full((train_edge_index.size(1),), 2, dtype=torch.long),
            torch.full((val_edge_index.size(1),), 1, dtype=torch.long),
            torch.full((test_edge_index.size(1),), 0, dtype=torch.long),
        ]
    )

    # Create data object with split information stored as attributes
    data = Data(x=features, edge_index=all_edges)
    data.split_labels = split_labels  # 2=train, 1=valid, 0=test
    data.has_predefined_split = True

    logger.debug(
        f"Loaded ogbl-ddi dataset with {data.num_nodes} nodes and {data.edge_index.size(1)} edges\n"
        f"Train edges: {train_edge_index.size(1)}, Val edges: {val_edge_index.size(1)}, Test edges: {test_edge_index.size(1)}"
    )

    return data, node_id_map


def get_graph_data(config: Config) -> Tuple[Data, dict]:
    """Unified dispatcher function to load graph data based on configuration.

    Routes to either CSV-based loading or ogbl-ddi loading based on config.graph.current_graph.

    Args:
        config (Config): Configuration object.

    Returns:
        Tuple[Data, dict]: A PyTorch Geometric Data object and node_id_map.
    """
    if config.graph.current_graph == "ogbl-ddi":
        logger.debug("Loading predefined ogbl-ddi graph")
        return _get_graph_data_from_ogbl_ddi(config)
    else:
        logger.debug(f"Loading CSV graph: {config.graph.current_graph}")
        file_path = config.graph.available_graphs[config.graph.current_graph]

        DDI_df = pd.read_csv(file_path, sep="\t").rename(columns={"Drug1": "src", "Drug2": "dst"})

        # shuffle the dataframe
        DDI_df = DDI_df.sample(frac=1, random_state=config.graph.seed_graph_sampling).reset_index(drop=True)

        return _get_graph_data_from_csv(DDI_df, config)