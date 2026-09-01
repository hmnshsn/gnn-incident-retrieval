# gnn-incident-retrieval

Research experiment testing whether GNNs over incident ontology graphs can
retrieve similar past incidents better than text-only baselines.

## Dataset

BPI Challenge 2014 (incident management process of a Dutch IT company).
Raw CSVs live under `data/raw/`.

## Layout

- `src/data_loader.py` — load + clean BPI 2014 CSVs
- `src/graph_builder.py` — convert tabular data to PyG `HeteroData`
- `src/baselines.py` — text-only and rule-based retrieval baselines
- `src/models.py` — GNN models
- `src/evaluate.py` — Hits@k / MRR computation
- `src/config.py` — experiment hyperparameters
- `scripts/` — download + experiment runners
- `notebooks/` — exploratory analysis

## Setup

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```
