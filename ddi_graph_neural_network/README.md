# ddi_graph_neural_network

This package provides the core Graph Neural Network (GNN) implementation for Drug-Drug Interaction (DDI) link prediction. It is built using PyTorch and PyTorch Geometric.

## Installation

Ensure you have a `pyproject.toml` set up in the root directory, then run:

```bash
uv pip install -e .
```

## Package Architecture

The package is strictly modularized into distinct operational areas:

### 1. `config.py` (Configuration)
Contains all the configurations and hyperparameter definitions using dataclasses. It handles:
- **Graph settings:** Which dataset to use, feature types (e.g., GPT embeddings, SMILES).
- **Training parameters:** Epochs, learning rate, random seeds.
- **Run settings:** Loss functions, negative sampling strategies, and balancing ratios.

### 2. `data_utils.py` (Data Pipeline)
Handles all data processing, graph construction, and feature alignment. 
- Loads raw edge data (interactions) and node features (drug embeddings).
- Maps drug IDs to graph node indices.
- Constructs the core PyTorch Geometric `Data` object containing `x` (node features) and `edge_index` (graph topology).

### 3. `model.py` (Model Architecture)
Defines the neural network structure.
- Contains the `Net` class, typically a multi-layer Graph Convolutional Network (GCN).
- **`encode()`:** Processes node features through GNN layers to generate low-dimensional node embeddings.
- **`decode()` / `decode_all()`:** Computes the probability of an edge existing between two nodes (usually via a dot-product predictor).

### 4. `train_model.py` (Training & Evaluation)
Manages the orchestration of the training loop and data splits.
- **Data Splitting:** Contains complex logic to split edges into train, validation, and test sets, including negative sampling (generating non-existent edges) or utilizing real negative edges.
- **Training Loop:** The `train()` function handles forward passes, loss computation (e.g., BCE or Focal Loss), and backpropagation.
- **Evaluation:** The `test()` and `get_metrics()` functions calculate key performance metrics like ROC-AUC and Average Precision (PR-AUC).