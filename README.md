# Detecting Drug Interactions with Machine Learning

This repository contains the codebase and empirical research for predicting Drug-Drug Interactions (DDIs) using Graph Neural Networks (GNNs) enriched with Large Language Model (LLM) textual embeddings.

## Project Architecture

The project workflow relies on two primary components:

1. **The Core Package (`ddi_graph_neural_network/`)**  
   An installable Python package containing the model architectures, data-processing pipelines, graph abstractions, and training loop logic.
   
2. **The Research Analysis (`report/`)**  
   The primary experimental artifact. The notebook `report/report.ipynb` imports the `ddi_graph_neural_network` core package to execute multi-component experiments, perform performance/error analysis based on ATC classes, and interactively represent the underlying embeddings and predictive behaviors.

### Auxiliary Folders
- `scripts/`: Command-line entrypoints to invoke batch model training.
- `tests/`: Unit test suite ensuring the reliability of the core package logic.
- `analysis/`, `data_preparation/`, `legacy/`: Notebooks and data-wrangling scripts previously used for parsing sources and evaluating isolated sub-processes.

## Quickstart

To reproduce the project and execute the `report/report.ipynb` notebook, follow these steps to set up the environment with `uv` and install the main package.

1) **Create and activate a virtual environment**:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2) **Install dependencies and the project package**:
Sync the environment dependencies and install the local `ddi_graph_neural_network` package in editable mode so it can be successfully imported inside the notebook.

```bash
uv sync
uv pip install -e .
```

*Note: Depending on your hardware, you may need to install specific PyTorch versions (e.g., CUDA-enabled) prior to syncing using: `uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0`*

3) **Execute the Report Notebook**:
You can now open `report/report.ipynb` in VS Code and select the newly created `.venv` as the notebook kernel, or run Jupyter directly from the terminal:

```bash
uv run jupyter notebook report/report.ipynb
```

## Adding Dependencies

This project uses `uv` connected to the `pyproject.toml` file for fast dependency management.

```bash
# Add a new dependency to pyproject.toml
uv add <package_name>

# Sync the changes to your local virtual environment
uv sync
```

---

## Command Line Usage

While the comprehensive experimental design runs out of `report/report.ipynb`, you can easily interface with the core package directly from the command line.

### Run Tests

Validate the internal operations by running the testing suite:

```bash
uv run pytest -q
```

### Run Model Training

Run the single standard end-to-end training procedure:

```bash
uv run python -m ddi_graph_neural_network.train_model
```


Run batch configuration loops and output results to a CSV log:

```bash
uv run python scripts/run_training.py
```
