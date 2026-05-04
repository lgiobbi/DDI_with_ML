# Reproducing the Research Report

This guide explains how to configure your system and environment to correctly run the `report.ipynb` notebook from end-to-end. 

## 1. Data Setup

The model and notebook rely on pre-computed graph files and feature embeddings.

**Getting the Data:**
- **If using the ETH VM:** The data provides can be found in the shared directory. Unzip the main data archive using `unzip /data/shared/ddi_with_ml_data.zip -d /data/shared/ddi_with_ml_data`.
- **Otherwise:** Please request the dataset directly by emailing `luca.giobbi@gmail.com`.

**Configuring the Data Path:**
The project dynamically targets the dataset folder using the `DDI_DATA_DIR` environment variable. By default, it expects data at `/data/giobbi`. If you downloaded or unzipped the data elsewhere, you must export this path before starting Jupyter or launching VS Code:
```bash
export DDI_DATA_DIR=/data/shared/ddi_with_ml_data
```
You can also hardcode your own directory as default in `config.py` if more convenient.

## 2. Environment & Package Setup

This project uses `uv` for fast dependency management. If you don't have it installed, install `uv` first.

Navigate to the **root folder of the project** (`DDI_with_ML/`) and follow these steps:

1. **Create and activate a virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   Sync the required dependencies defined in `pyproject.toml`.
   ```bash
   uv sync
   ```

3. **Install the core package:**
   The `report.ipynb` heavily imports from the core scripts. You must install the main `ddi_graph_neural_network` package (the `ddi_with_ml` logic) in editable mode.
   ```bash
   uv pip install -e .
   ```

## 3. Running the Notebook

After setup, open `report.ipynb` (e.g., inside VS Code or via Jupyter Server).

**Important Checklist:**
- **Kernel Selection:** Ensure your IDE/Jupyter uses the `.venv` environment you just created.
- **Environment Variable:** Verify `DDI_DATA_DIR` is active in the environment your editor or server was launched from.
- **Logging Configuration:** To see detailed output of data routing, graph sizes, dropping sequences, and training steps, set the python logger to `DEBUG` at the start of your notebook:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Optionally, to target just the core package logs:
logging.getLogger("ddi_graph_neural_network").setLevel(logging.DEBUG)
```

Run the notebook from top to bottom. Results and visualizations will automatically output to the `report_outputs/` directory inside this folder.