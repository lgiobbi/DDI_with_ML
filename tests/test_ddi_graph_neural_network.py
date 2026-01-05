import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from ddi_graph_neural_network.config import Config
from ddi_graph_neural_network.data_utils import (
    get_features_and_edges_constant,
    get_graph_data,
    get_node_id_map,
    intersect_graph_and_embeddings,
    match_embeddings_to_graph,
)
from ddi_graph_neural_network.model import Net
from ddi_graph_neural_network.train_model import (
    data_split_with_labels,
    prepare_train_labels,
    run_training,
)
import ddi_graph_neural_network.train_model as tm

# UNIT TESTS

def _toy_ddi_df(num_pos: int = 6, num_neg: int = 6) -> pd.DataFrame:
    # 8 nodes, generate disjoint pos/neg edge sets deterministically
    nodes = [f"D{i}" for i in range(8)]

    pos_edges = []
    for i in range(num_pos):
        a = nodes[i % 8]
        b = nodes[(i + 1) % 8]
        pos_edges.append((a, b, 1))

    neg_edges = []
    for i in range(num_neg):
        a = nodes[(i + 2) % 8]
        b = nodes[(i + 4) % 8]
        if a == b:
            b = nodes[(i + 5) % 8]
        neg_edges.append((a, b, 0))

    df = pd.DataFrame(pos_edges + neg_edges, columns=["src", "dst", "label"])
    return df


def _write_toy_embeddings_tsv(path, drug_ids) -> str:
    # get_graph_data reads via: pd.read_csv(sep='\t', index_col=0)
    # so we provide a dummy first column to become the index.
    df = pd.DataFrame(
        {
            "idx": np.arange(len(drug_ids)),
            "Drug ID": drug_ids,
            "f1": np.linspace(0.1, 1.0, len(drug_ids)).astype(float),
            "f2": np.linspace(1.0, 2.0, len(drug_ids)).astype(float),
        }
    )
    file_path = str(path / "emb.tsv")
    df.to_csv(file_path, sep="\t", index=False)
    return file_path


# Test that get_node_id_map indexes all unique src/dst nodes.
def test_get_node_id_map_contains_all_nodes():
    ddi = pd.DataFrame({"src": ["A", "B", "C"], "dst": ["B", "C", "A"]})
    node_id_map = get_node_id_map(ddi)

    assert set(node_id_map.keys()) == {"A", "B", "C"}
    assert set(node_id_map.values()) == set(range(3))


# Test that embeddings are aligned to graph and missing drugs get dummy rows.
def test_match_embeddings_to_graph_adds_missing_drugs():
    ddi = pd.DataFrame({"src": ["A", "B"], "dst": ["B", "C"]})
    emb = pd.DataFrame({"Drug ID": ["A", "B"], "f": [0.5, 0.7]})

    aligned, node_id_map = match_embeddings_to_graph(ddi, emb, drug_id_col="Drug ID")

    assert list(aligned.index) == ["A", "B", "C"]
    assert node_id_map["A"] == 0 and node_id_map["C"] == 2
    assert float(aligned.loc["C", "f"]) == 1.0


# Test that edges/drugs not present in embeddings are dropped from the graph.
def test_intersect_graph_and_embeddings_drops_edges_not_in_embeddings():
    ddi = pd.DataFrame({"src": ["A", "B", "X"], "dst": ["B", "C", "A"], "label": [1, 1, 0]})
    emb = pd.DataFrame({"Drug ID": ["A", "B", "C"], "f": [0.1, 0.2, 0.3]})

    ddi2, emb2, node_id_map = intersect_graph_and_embeddings(ddi, emb, drug_id_col="Drug ID")

    assert set(ddi2["src"]).union(set(ddi2["dst"])) <= {"A", "B", "C"}
    assert "X" not in node_id_map
    assert set(emb2.index) == {"A", "B", "C"}


# Test constant feature construction and edge_index shape from a tiny graph.
def test_get_features_and_edges_constant_shapes_and_values():
    ddi = pd.DataFrame({"src": ["A", "B"], "dst": ["B", "C"]})
    features, edge_index = get_features_and_edges_constant(ddi, feature_value=2.0, feature_dim=3)

    assert features.shape == (3, 3)
    assert torch.allclose(features, torch.full((3, 3), 2.0))
    assert edge_index.shape == (2, 2)


# Test get_graph_data with __ONES__ features when negatives are ignored in labels.
def test_get_graph_data_ones_with_labels_take_negative_samples_false():
    ddi = _toy_ddi_df(num_pos=5, num_neg=3)
    cfg = Config()
    cfg.graph.feature = "__ONES__"
    cfg.run.take_negative_samples = False
    cfg.run.balanced_labels = True

    data, node_id_map = get_graph_data(ddi, cfg)

    # should keep only positives (balanced down to min(pos, neg)) and set y=None
    assert data.y is None
    assert data.x.shape[0] == len(node_id_map)
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] == 3


# Test get_graph_data with __ONES__ features when positives/negatives are balanced.
def test_get_graph_data_ones_with_labels_take_negative_samples_true():
    ddi = _toy_ddi_df(num_pos=5, num_neg=3)
    cfg = Config()
    cfg.graph.feature = "__ONES__"
    cfg.run.take_negative_samples = True
    cfg.run.balanced_labels = True

    data, node_id_map = get_graph_data(ddi, cfg)

    assert data.y is not None
    assert data.edge_index.shape[1] == 6
    assert data.y.numel() == 6
    assert int((data.y == 1).sum().item()) == 3
    assert int((data.y == 0).sum().item()) == 3
    assert data.x.shape[0] == len(node_id_map)


# Test get_graph_data reading real embeddings from a TSV file.
def test_get_graph_data_from_embeddings_reads_tsv(tmp_path, monkeypatch):
    ddi = _toy_ddi_df(num_pos=4, num_neg=4)
    drug_ids = sorted(set(ddi["src"]).union(set(ddi["dst"])))
    emb_path = _write_toy_embeddings_tsv(tmp_path, drug_ids)

    cfg = Config()
    cfg.graph.feature = "DESC_GPT"
    cfg.run.take_negative_samples = True

    # Save the original pd.read_csv before monkeypatching
    real_read_csv = pd.read_csv

    def fake_read_csv(path, sep="\t", index_col=0):
        # get_graph_data will call this; we ignore the path and load our test file
        assert sep == "\t"
        assert index_col == 0
        return real_read_csv(emb_path, sep=sep, index_col=index_col)

    monkeypatch.setattr("ddi_graph_neural_network.data_utils.pd.read_csv", fake_read_csv)

    data, node_id_map = get_graph_data(ddi, cfg)

    assert data.x.dtype == torch.float32
    assert data.x.shape[0] == len(node_id_map)
    assert data.x.shape[1] == 2
    assert int(data.edge_index.max().item()) < data.x.shape[0]
    assert data.y is not None


# Test full Net forward/backward pass on toy data for gradient flow.
def test_net_forward_and_backward_smoke():
    torch.manual_seed(0)

    ddi = _toy_ddi_df(num_pos=4, num_neg=4)
    cfg = Config()
    cfg.graph.feature = "__ONES__"
    cfg.run.take_negative_samples = True
    cfg.run.balanced_labels = True
    data, _ = get_graph_data(ddi, cfg)

    # Build supervision edges/labels for link prediction
    edge_label_index, edge_label = prepare_train_labels(data.edge_index, num_nodes=data.x.size(0))
    model = Net(in_channels=data.x.size(1), hidden_channels=8, out_channels=8)

    out = model(data.x, data.edge_index, edge_label_index)
    assert out.shape == (edge_label_index.size(1),)

    loss = torch.nn.BCEWithLogitsLoss()(out, edge_label)
    loss.backward()

    grad_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sum += float(p.grad.abs().sum().item())
    assert grad_sum > 0.0


# Test that prepared train labels include positives/negatives and no self-loops.
def test_prepare_train_labels_contains_negatives_and_no_self_loops():
    torch.manual_seed(0)

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_label_index, edge_label = prepare_train_labels(edge_index=edge_index, num_nodes=5)

    assert edge_label_index.shape[0] == 2
    assert edge_label.shape[0] == edge_label_index.shape[1]
    assert (edge_label == 1).any().item()
    assert (edge_label == 0).any().item()

    u = edge_label_index[0]
    v = edge_label_index[1]
    assert bool((u != v).all().item())


# Test that data_split_with_labels produces supervision edges for all splits.
def test_data_split_with_labels_outputs_supervision_edges():
    torch.manual_seed(0)

    ddi = _toy_ddi_df(num_pos=10, num_neg=10)
    cfg = Config()
    cfg.graph.feature = "__ONES__"
    cfg.run.take_negative_samples = True
    cfg.run.balanced_labels = True
    data, _ = get_graph_data(ddi, cfg)

    train_data, val_data, test_data = data_split_with_labels(data)

    for split in (train_data, val_data, test_data):
        assert hasattr(split, "edge_label_index")
        assert hasattr(split, "edge_label")
        assert split.edge_label_index.shape[0] == 2
        assert split.edge_label.numel() == split.edge_label_index.shape[1]
        # expect both classes to exist (given we split pos/neg separately)
        assert int((split.edge_label == 1).sum().item()) > 0
        assert int((split.edge_label == 0).sum().item()) > 0







# LINKED FUNCTIONALITY TESTS

# Test that run_training selects best validation epoch and early-stops using mocks.
def test_run_training_uses_best_val_auc_and_early_stopping(monkeypatch):
    cfg = Config()
    cfg.training.epochs = 5
    cfg.training.patience = 1
    cfg.training.learning_rate = 1e-3
    cfg.training.lr_lambda = 0.9

    base_data = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )

    fake_train = Data()
    fake_val = Data()
    fake_test_data = Data()

    def fake_data_split_with_labels(data):
        return fake_train, fake_val, fake_test_data

    monkeypatch.setattr(tm, "data_split_with_labels", fake_data_split_with_labels)

    val_aucs = [0.5, 0.7, 0.6, 0.4]
    test_labels_seq = [
        np.array([0, 1], dtype=float),
        np.array([0, 1], dtype=float),
        np.array([0, 1], dtype=float),
        np.array([0, 1], dtype=float),
    ]
    test_scores_seq = [
        np.array([0.3, 0.7], dtype=float),
        np.array([0.1, 0.9], dtype=float),
        np.array([0.2, 0.8], dtype=float),
        np.array([0.4, 0.6], dtype=float),
    ]

    call_count = {"n": 0}

    def fake_test(model, data_split):
        epoch_idx = call_count["n"] // 2
        if call_count["n"] % 2 == 0:
            result = (val_aucs[epoch_idx], np.array([], dtype=float), np.array([], dtype=float))
        else:
            result = (0.0, test_labels_seq[epoch_idx], test_scores_seq[epoch_idx])
        call_count["n"] += 1
        return result

    monkeypatch.setattr(tm, "test", fake_test)

    def fake_train_fn(model, optimizer, criterion, scheduler, train_data):
        return torch.tensor(0.123, dtype=torch.float32)

    monkeypatch.setattr(tm, "train", fake_train_fn)

    device = torch.device("cpu")
    model, label, scores, returned_test_data = tm.run_training(
        cfg,
        base_data,
        device,
    )

    assert isinstance(model, Net)
    assert isinstance(returned_test_data, Data)
    assert np.array_equal(label, test_labels_seq[1])
    assert np.array_equal(scores, test_scores_seq[1])
    assert call_count["n"] == 6


# Test that main wires Config, get_graph_data, run_training, and get_metrics together via mocks.
def test_main_pipeline_with_mocks(monkeypatch):
    cfg = Config()
    cfg.graph.current_graph = "MOCK_GRAPH"
    cfg.graph.available_graphs[cfg.graph.current_graph] = "mock_path.tsv"
    cfg.graph.feature = "GPT+Desc"
    cfg.run.take_negative_samples = True

    toy_ddi = _toy_ddi_df(num_pos=2, num_neg=2).rename(columns={"src": "Drug1", "dst": "Drug2"})

    def fake_read_csv(path, sep="\t"):
        assert path == cfg.graph.available_graphs[cfg.graph.current_graph]
        assert sep == "\t"
        return toy_ddi

    monkeypatch.setattr(tm.pd, "read_csv", fake_read_csv)

    called = {"get_graph_data": 0, "run_training": 0}

    def fake_get_graph_data(ddi_df, config):
        called["get_graph_data"] += 1
        assert config is cfg
        data = Data(
            x=torch.ones((4, 1), dtype=torch.float32),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        )
        node_id_map = {f"D{i}": i for i in range(4)}
        return data, node_id_map

    monkeypatch.setattr(tm, "get_graph_data", fake_get_graph_data)

    dummy_model = object()
    dummy_label = np.array([0.0, 1.0], dtype=float)
    dummy_scores = np.array([0.2, 0.8], dtype=float)
    dummy_test_data = Data()

    def fake_run_training(config, graph_data, device):
        called["run_training"] += 1
        assert config is cfg
        return dummy_model, dummy_label, dummy_scores, dummy_test_data

    monkeypatch.setattr(tm, "run_training", fake_run_training)

    def fake_get_metrics(label, scores):
        assert np.array_equal(label, dummy_label)
        assert np.array_equal(scores, dummy_scores)
        return {"AUC": 0.9, "PR_AUC": 0.8}

    monkeypatch.setattr(tm, "get_metrics", fake_get_metrics)

    results = tm.main(cfg)

    assert called["get_graph_data"] == 1
    assert called["run_training"] == 1
    assert results["config"] is cfg
    assert results["model"] is dummy_model
    assert results["label"] is dummy_label
    assert results["test_scores"] is dummy_scores
    assert results["test_data"] is dummy_test_data
    assert results["metrics"]["AUC_mean"] == 0.9
    assert results["metrics"]["PR_AUC_mean"] == 0.8


@pytest.mark.slow
# Slow smoke test that runs a short real training loop on CPU.
def test_run_training_smoke_cpu_small_epochs():
    torch.manual_seed(0)
    np.random.seed(0)

    ddi = _toy_ddi_df(num_pos=12, num_neg=12)
    cfg = Config()
    cfg.graph.feature = "__ONES__"
    cfg.run.take_negative_samples = True
    cfg.run.balanced_labels = True
    cfg.training.seed = 0
    cfg.training.epochs = 4
    cfg.training.patience = 2
    cfg.training.learning_rate = 1e-3
    cfg.training.lr_lambda = 0.99
    data, _ = get_graph_data(ddi, cfg)

    device = torch.device("cpu")
    model, label, scores, test_data = run_training(
        cfg,
        data,
        device,
    )

    assert isinstance(model, Net)
    assert isinstance(label, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert label.shape == scores.shape
    assert hasattr(test_data, "edge_label")
