# GNN Incident Retrieval

Graph Neural Networks for incident-to-incident retrieval in IT Service Management.

## Datasets

| Dataset             | Incidents | CIs   | Resolution Codes | Categories | Subcategories  | Text Fields                    |
| ------------------- | --------- | ----- | ---------------- | ---------- | -------------- | ------------------------------ |
| BPI 2014 (Rabobank) | 46,146    | 2,794 | 14               | 4          | —             | None                           |
| ServiceNow Internal | 59,151    | 7,140 | 25               | 36         | 154 (57% null) | short_description, description |

## Model Architecture

**GNN: Heterogeneous GraphSAGE (HeteroSAGE)**

2-layer GraphSAGE convolutions wrapped in PyTorch Geometric's HeteroConv with mean aggregation across edge types. Each edge type (incident-to-CI, incident-to-category, CI-to-subtype, etc.) has its own SAGEConv layer. Non-incident nodes (CI, category, subtype) use learnable nn.Embedding tables. Incident nodes use TF-IDF features over available text fields. For closure-code prediction, a linear classification head produces logits. For incident retrieval, embeddings are extracted from the pre-classifier hidden layer and ranked by cosine similarity.

**Graph structure:**
- Node types: incident, ci, ci_subtype, category (BPI 2014 also includes ci_type and ci_cby)
- Edge types: bidirectional edges between incidents and their CI, category, and subtype nodes
- No target leakage: closure codes are classification targets only, not graph nodes

## Baseline Methods

### Closure Code Prediction (Exp02)

| Method | Description |
|--------|-------------|
| Random | Ranks closure codes in random order for each incident |
| Category-Match | For each incident, ranks closure codes by frequency within its category (e.g., if category "Hardware" most often resolves as "Solved Permanently", that code ranks first) |
| CI-Majority | For each incident, ranks closure codes by frequency for its specific CI. If CI "ServerX" resolved as "Hardware Fix" 80% of the time, that code ranks first. Falls back to global frequency for unseen CIs |
| Text-Similarity | Computes TF-IDF similarity between the test incident's text and training incidents, ranks closure codes by weighted vote from most similar incidents |

### Incident-to-Incident Retrieval (Exp03)

| Method | Description |
|--------|-------------|
| Random | Assigns random similarity scores between all query-candidate pairs |
| CI-Match | Scores 1.0 if query and candidate share the same CI, 0.0 otherwise |
| CI-Subtype | Scores 1.0 for same CI, 0.5 for same CI subtype but different CI, 0.0 otherwise |
| Category | Scores 1.0 if query and candidate share the same category, 0.0 otherwise |

## Exp02: Closure Code Prediction

### BPI 2014

| Method          | MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
| --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Random          | 0.233           | 0.071           | 0.216           | 0.362           | 0.721           |
| Category-Match  | 0.578           | 0.363           | 0.704           | 0.872           | 1.000           |
| Text-Similarity | 0.578           | 0.350           | 0.743           | 0.928           | 1.000           |
| CI-Majority     | 0.686           | 0.521           | 0.805           | 0.912           | 0.987           |
| **GNN**   | **0.702** | **0.540** | **0.821** | **0.925** | **1.000** |

### ServiceNow Internal

| Method          | MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
| --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Random          | 0.150           | 0.039           | 0.115           | 0.196           | 0.392           |
| Category-Match  | 0.505           | 0.296           | 0.652           | 0.785           | 0.916           |
| Text-Similarity | 0.525           | 0.338           | 0.618           | 0.795           | 0.942           |
| CI-Majority     | 0.578           | 0.409           | 0.691           | 0.788           | 0.907           |
| **GNN**   | **0.614** | **0.443** | **0.741** | **0.863** | **0.953** |

## Exp03: Incident-to-Incident Retrieval (Graded Relevance)

Relevance: 3 = same CI + same resolution code, 2 = same CI, 1 = same subtype + same resolution code, 0 = otherwise.

### BPI 2014

| Method        | nDCG@1          | nDCG@3          | nDCG@5          | nDCG@10         | nDCG@20         | MAP             | MRR             |
| ------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Random        | 0.021           | 0.023           | 0.024           | 0.023           | 0.024           | 0.083           | 0.200           |
| Category      | 0.019           | 0.021           | 0.026           | 0.027           | 0.027           | 0.100           | 0.214           |
| CI-Match      | 0.565           | 0.603           | 0.610           | 0.622           | 0.642           | 0.360           | 0.938           |
| CI-Subtype    | 0.570           | 0.612           | 0.623           | 0.637           | 0.661           | 0.549           | 0.953           |
| **GNN** | **0.643** | **0.641** | **0.642** | **0.643** | **0.648** | **0.441** | **0.944** |

### ServiceNow Internal

| Method        | nDCG@1          | nDCG@3          | nDCG@5          | nDCG@10         | nDCG@20         | MAP             | MRR             |
| ------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Random        | 0.022           | 0.022           | 0.022           | 0.023           | 0.024           | 0.067           | 0.177           |
| Category      | 0.043           | 0.070           | 0.078           | 0.087           | 0.096           | 0.142           | 0.275           |
| CI-Match      | 0.557           | 0.572           | 0.578           | 0.582           | 0.589           | 0.354           | 0.876           |
| CI-Subtype    | 0.579           | 0.595           | 0.600           | 0.608           | 0.619           | 0.473           | 0.910           |
| **GNN** | **0.591** | **0.591** | **0.586** | **0.583** | **0.579** | **0.380** | **0.879** |

### Subtype Edge Ablation (ServiceNow)

| Setting | Subtype Null Rate | GNN nDCG@1 | GNN nDCG@3 | GNN nDCG@5 | GNN nDCG@10 | GNN nDCG@20 | GNN MAP | GNN MRR |
|---------|-------------------|------------|------------|------------|-------------|-------------|---------|---------|
| none | 100% | 0.572 | 0.583 | 0.581 | 0.582 | 0.584 | 0.373 | 0.890 |
| **raw** | **56.86%** | **0.596** | **0.588** | **0.586** | **0.582** | **0.579** | **0.380** | **0.879** |
| imputed | 40.61% | 0.583 | 0.587 | 0.585 | 0.583 | 0.589 | 0.425 | 0.881 |

Raw subtypes (57% null) give best top-1 ranking (nDCG@1 = 0.596). Imputation improves MAP (+0.045 over raw) at a cost to nDCG@1. No subtypes is worst at top-1. For ZTSD (where the agent picks top-1/top-3 retrieved incidents), raw subtype mode is preferred.

## Exp04 — Text Signal Fusion (SN)

Late fusion combines precomputed sentence embeddings (all-MiniLM-L6-v2, 384-dim) with GNN embeddings post-training. Alpha controls the weight: alpha=1.0 is GNN-only, alpha=0.0 is text-only.

Alpha sweep (30 epochs, raw subtype):

| Alpha | nDCG@1 | nDCG@5 | nDCG@10 | MAP | MRR |
|-------|--------|--------|---------|------|------|
| Text-only (0.0) | 0.357 | 0.348 | 0.343 | 0.175 | 0.638 |
| 0.5 | 0.569 | 0.551 | 0.543 | 0.344 | 0.846 |
| 0.6 | 0.607 | 0.592 | 0.587 | 0.375 | 0.873 |
| 0.7 | 0.625 | 0.614 | 0.607 | 0.383 | 0.884 |
| **0.8** | **0.629** | **0.614** | **0.608** | **0.383** | **0.885** |
| 0.85 | 0.628 | 0.613 | 0.606 | 0.382 | 0.883 |
| 0.9 | 0.628 | 0.611 | 0.604 | 0.381 | 0.882 |
| 0.95 | 0.624 | 0.609 | 0.602 | 0.381 | 0.880 |
| GNN-only (1.0) | 0.594 | 0.588 | 0.582 | 0.380 | 0.879 |

Best alpha: 0.8 (80% GNN, 20% text). Text acts as a tiebreaker for top-1 retrieval.

## Exp05 — Cold-Start Analysis (SN)

Test set split by CI visibility in training: 7284 seen, 1449 unseen (16.6%).

| Group | Method | nDCG@1 | MAP | MRR |
|-------|--------|--------|------|------|
| Seen | Text-only | 0.393 | 0.193 | 0.705 |
| Seen | GNN-only | 0.657 | 0.404 | 0.962 |
| Seen | Late fusion (0.8) | **0.690** | **0.407** | **0.967** |
| Unseen | Text-only | 0.174 | 0.086 | 0.301 |
| Unseen | GNN-only | 0.281 | 0.259 | 0.459 |
| Unseen | Late fusion (0.8) | **0.326** | **0.262** | **0.474** |

Cold-start degradation: GNN nDCG@1 drops 57% (0.657 to 0.281) for unseen CIs. Fusion lift is larger for unseen (+0.045) than seen (+0.033).

## Exp06 — Early Fusion Ablation (SN)

Text embeddings as GNN input features instead of post-hoc fusion. Early stopping (patience=10).

| Features | All nDCG@1 | Seen nDCG@1 | Unseen nDCG@1 | All MAP | All MRR |
|----------|-----------|-------------|---------------|---------|---------|
| TF-IDF (baseline) | 0.588 | 0.650 | 0.276 | 0.380 | 0.880 |
| Sentence embeddings | 0.626 | 0.686 | 0.326 | 0.387 | 0.886 |
| TF-IDF + sentence (concat) | 0.624 | 0.681 | **0.336** | **0.389** | **0.888** |
| Late fusion (0.8) | **0.629** | **0.690** | 0.326 | 0.383 | 0.885 |

Late fusion matches early fusion on nDCG@1. Concat wins MAP and cold-start nDCG@1.

## Key Findings

1. CI identity is the dominant signal for incident resolution prediction across both datasets.
2. GNN adds value via CI type/subtype propagation — lift is larger on the richer SN dataset (+0.036 MRR) than BPI 2014 (+0.016 MRR) for closure-code prediction.
3. For incident retrieval, GNN achieves best nDCG@1 on both datasets (best top-1 ranking quality).
4. On SN data, GNN retrieval degrades past top positions due to 57% null subcategories limiting graph propagation. CI-Subtype baseline wins on MAP and deeper nDCG cutoffs.
5. Cold-start gap on BPI 2014: seen CIs achieve 0.697 MRR vs unseen CIs at 0.522 MRR — this is where GNN helps most.

## Setup

```bash
cd ~/hmnshpl/Graphs/gnn-incident-retrieval
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```
## Reproducing Results

All experiments use seed 42. BPI 2014 data is downloaded via script; ServiceNow data is not publicly available.

```bash
# BPI 2014
bash scripts/01_download_data.sh
uv run python scripts/02_run_baselines.py
uv run python scripts/03_run_gnn_closure_code.py
uv run python scripts/05_run_incident_retrieval.py

# ServiceNow (requires internal dataset in data/from_praison/)
uv run python scripts/07_run_sn_exp02_closure_code.py
uv run python scripts/08_run_sn_exp03_retrieval.py --subtype raw
uv run python scripts/08_run_sn_exp03_retrieval.py --subtype imputed
uv run python scripts/08_run_sn_exp03_retrieval.py --subtype none
```

## What's next

1. Adaptive alpha — binary split (seen CIs: alpha=0.9, unseen CIs: alpha=0.6) to exploit the cold-start finding
2. Per-incident win/loss analysis — quantify what fraction of test incidents improve vs degrade under fusion
3. Incident-to-KB retrieval — pending KB number field fix from data provider
4. Cold-start analysis on BPI 2014 — replicate SN cold-start finding on the second dataset
