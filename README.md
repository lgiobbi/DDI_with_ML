# DDI with ML

Drug–drug interaction (DDI) prediction experiments using graph neural networks.

The main code artifact of this repository is the installable Python package in `ddi_graph_neural_network/`.
It contains the model, training loop, and graph/data utilities used by the scripts and tests.

## Repository structure

- `ddi_graph_neural_network/`: main package (model + training + data utilities)
- `scripts/`: command-line style entrypoints (e.g., batch training runs)
- `tests/`: unit tests
- `analysis/`, `data_preparation/`, `legacy/`: notebooks and older experiments

## Requirements

- Python: >= 3.10 (see `pyproject.toml`)
- PyTorch + PyTorch Geometric

### Install

Create and activate a virtual environment, then install dependencies.

This project uses `uv` for environment and dependency management.

1) Create a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

2) Install PyTorch for your system (CPU or CUDA). Example (CPU):

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0
```

3) Install the remaining dependencies and the package:

```bash
uv pip install -r requirements.txt
uv pip install -e .
```

## Run training

The simplest “real” training entrypoint is:

```bash
uv run python -m ddi_graph_neural_network.train_model
```

Batch runs that write a CSV summary live in:

```bash
uv run python scripts/run_training.py
```

## Run tests

```bash
uv run pytest -q
```
