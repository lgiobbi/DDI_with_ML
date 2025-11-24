from typing import Tuple
import numpy as np
import pandas as pd
import torch

KEPT_PERC_NOT_IN_GRAPH = 0.0  # Percentage of drugs not in graph to keep


def get_node_id_map(DDI_graph: pd.DataFrame) -> dict:
    """Get a mapping from drug IDs to node indices.

    Args:
        DDI_graph (pd.DataFrame): Drug-drug interaction graph.

    Returns:
        dict: Mapping from drug IDs to node indices.
    """
    DrugIDs_in_graph = np.unique(DDI_graph[["src", "dst"]].values)
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


def match_embeddings_to_graph(
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


def intersect_graph_and_embeddings(
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


def get_features_and_edges(
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

    edge_index = DDI_df[["src", "dst"]].map(lambda id: map_node_id(node_id_map, id)).to_numpy()
    edge_index = torch.tensor(edge_index).t().contiguous()

    # DDI_graph = np.vstack((DDI_graph, DDI_graph[:, ::-1]))  # Make bidirectional
    return features, edge_index


def get_features_and_edges_constant(
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
    node_id_map = get_node_id_map(DDI_df)
    num_nodes = len(node_id_map)
    features = torch.full((num_nodes, feature_dim), float(feature_value), dtype=torch.float32)

    edge_index = DDI_df[["src", "dst"]].map(lambda id: map_node_id(node_id_map, id)).to_numpy()
    edge_index = torch.tensor(edge_index).t().contiguous()

    return features, edge_index
