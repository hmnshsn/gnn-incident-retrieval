# GNN Incident Retrieval

Graph Neural Networks for incident-to-incident retrieval in IT Service Management.

## Datasets

| Dataset | Incidents | CIs | Resolution Codes | Categories | Subcategories | Features |
|---------|-----------|-----|------------------|------------|---------------|----------|
| BPI 2014 (Rabobank) | 46,146 | 2,794 | 14 | 4 | — | categorical fields |
| ServiceNow Internal | 59,151 | 7,140 | 25 | 36 | 154 (57% null) | categorical fields, short_description |

BPI 2014 has no free-text incident features. Both datasets use SentenceTransformer embeddings for categorical feature text; ServiceNow also has precomputed short-description embeddings.

## Model Architecture

**GNN: heterogeneous GraphSAGE (HeteroSAGE)**

Two GraphSAGE convolution layers wrapped in PyTorch Geometric's `HeteroConv`, with mean aggregation across edge types. Each relation has its own `SAGEConv` layer. Non-incident nodes (CI, category, subtype, and related entities) use learnable embedding tables. Incident nodes use categorical SentenceTransformer features and optional short-description embeddings. A linear head predicts closure codes. Retrieval uses pre-classifier incident embeddings ranked by cosine similarity.

**Graph structure:**

- Node types include `incident`, `ci`, `ci_subtype`, and `category`; BPI 2014 also includes `ci_type` and `ci_cby`.
- Edges are bidirectional between incidents and CI, category, subtype, and related entity nodes.
- Optional ServiceNow `assignment_group` nodes are supported.
- No target leakage: closure codes are classification targets only, not graph nodes.

## File Structure

- `scripts/09_encode_sn_text.py` — precompute ServiceNow short-description embeddings
- `scripts/10_extract_sn_gnn_embeddings.py` — extract GNN embeddings and split indices
- `scripts/11_run_sn_text_retrieval.py` — text-only, GNN-only, and fusion retrieval
- `scripts/12_run_sn_cold_start_analysis.py` — ServiceNow seen vs unseen CI analysis
- `scripts/13_run_sn_early_fusion.py` — early-fusion feature ablation
- `scripts/15_run_bpi_cold_start_analysis.py` — BPI 2014 cold-start replication
- `scripts/23_sweep_agent.py` — W&B sweep configuration and agent
- `scripts/24_classification_diagnostics.py` — feature-only classification diagnostics
- `src/graph_builder.py` — heterogeneous graph construction
- `src/models.py` — GraphSAGE model, training, and embedding extraction
- `src/relevance.py` — graded retrieval relevance definitions
- `src/sweep_runner.py` — shared sweep training and retrieval evaluation
- `src/text_encoder.py` — sentence embedding encoding and caching

## Baseline Methods

### Closure Code Prediction

- **Random** — random closure-code ranking.
- **Category-Match** — closure-code frequency within category.
- **CI-Majority** — closure-code frequency for the CI, with global fallback for unseen CIs.
- **Text-Similarity** — closure-code ranking from similar training incidents.

### Incident-to-Incident Retrieval

- **Random** — random query-candidate scores.
- **CI-Match** — score 1.0 for shared CI, otherwise 0.0.
- **CI-Subtype** — score 1.0 for shared CI, 0.5 for shared subtype across CIs.
- **Category** — score 1.0 for shared category, otherwise 0.0.

## Definition E: Problem-Similarity Retrieval

Definition E grades retrieval by problem similarity without closure-code gating: same CI = 3, same subtype = 2, same category = 1.

| Method | Seen nDCG@1 | Unseen nDCG@1 |
|--------|-------------|---------------|
| Raw text embeddings | 0.534 | 0.387 |
| Raw categorical features | 0.503 | 0.775 |
| Raw concat features | 0.603 | 0.627 |
| GNN epoch 0 (no training) | 0.939 | 0.801 |
| GNN + CE loss (best epoch) | 0.910 | 0.790 |
| GNN + Triplet loss (best epoch) | 0.924 | 0.799 |

Training on closure codes degrades unseen retrieval from epoch 1 onward. Confirmed on ServiceNow (58k incidents) and BPI 2014 (46k incidents).

## Path B: Resolution Prediction

Train GNN/MLP to predict `u_resolution` embeddings (384-dim) instead of closure codes. Evaluate by retrieving nearest train incidents in resolution embedding space.

**4-config comparison (seed 42, 50 epochs, early stopping patience 10):**

| Loss | Head | Model | Res Cosine | Top-1 Code | Top-5 Code |
|------|------|-------|------------|------------|------------|
| -- | -- | Text-only | 0.480 | 0.392 | 0.679 |
| cosine | linear | MLP | 0.648 | 0.378 | 0.607 |
| cosine | linear | GNN | 0.645 | 0.414 | 0.638 |
| cosine | proj | MLP | 0.642 | 0.378 | 0.599 |
| cosine | proj | GNN | 0.644 | 0.417 | 0.643 |
| infonce | linear | MLP | 0.391 | 0.384 | 0.643 |
| infonce | linear | GNN | 0.400 | 0.434 | 0.683 |
| infonce | proj | MLP | 0.378 | 0.370 | 0.631 |
| infonce | proj | GNN | 0.393 | 0.429 | 0.682 |

**3-seed confirmation (InfoNCE + linear, seeds 42/1/123):**

| Model | Res Cosine | Top-1 Code Match | Top-5 Code Match |
|-------|------------|------------------|------------------|
| Text-only | 0.480 | 0.392 | 0.679 |
| MLP | 0.391 +/- 0.001 | 0.382 +/- 0.002 | 0.641 +/- 0.002 |
| GNN | 0.395 +/- 0.004 | 0.430 +/- 0.005 | 0.678 +/- 0.003 |

Key findings:
- Projection head adds nothing over linear. Clean negative result.
- InfoNCE and cosine loss optimize different objectives: cosine maximizes resolution semantic similarity; InfoNCE maximizes retrieval discrimination.
- GNN beats MLP on closure code match across all configs (+0.036 to +0.059 top-1). Graph structural bias through CI nodes contributes real signal.
- InfoNCE + GNN + linear is the best config: 0.430 top-1 code match (+9.7% relative over text-only).

### Hybrid Retrieval (Text + GNN Resolution Fusion)

Fuse short_description text similarity with GNN-predicted resolution similarity:

```text
score = alpha * gnn_resolution_sim + (1 - alpha) * text_description_sim
```

Alpha tuned on val set (best alpha: 0.8-0.9 across seeds).

**3-seed results (seeds 42/1/123, 50 epochs, early stopping patience 10):**

| Model | Top-1 Code Match | Top-5 Code Match |
|-------|------------------|------------------|
| Text-only | 0.393 | 0.678 |
| GNN-only (InfoNCE) | 0.431 +/- 0.006 | 0.681 +/- 0.006 |
| Hybrid (alpha=0.8-0.9) | 0.445 +/- 0.007 | 0.685 +/- 0.004 |

**Val alpha sweep (mean top-1 code match across 3 seeds):**

| Alpha | Val Top-1 |
|-------|-----------|
| 0.0 (text-only) | 0.440 |
| 0.2 | 0.466 |
| 0.5 | 0.471 |
| 0.7 | 0.473 |
| 0.8 | 0.477 |
| 0.9 | 0.479 |
| 1.0 (GNN-only) | 0.473 |

Hybrid achieves best top-1 code match (+13.2% relative over text-only) and best top-5 code match simultaneously.

## Resolution Coverage (ServiceNow, Seen CIs)

| Metric | Value |
|--------|-------|
| Top-1 closure code match | 48.8% |
| Top-3 closure code match | 69.5% |
| Top-5 closure code match | 75.1% |
| Correct resolution exists in CI set | 92.0% |

## Key Findings

1. Epoch-0 GNN retrieval is strong despite random model weights.
2. Graph topology is the primary value driver; message passing exposes CI, subtype, and category structure.
3. Closure-code training improves classification but hurts unseen retrieval under Definition E.
4. The correct closure code is often available among same-CI training incidents, but selecting it remains difficult.
5. Results hold across ServiceNow and BPI 2014, supporting a topology-first retrieval baseline.

## What's Next

1. Per-incident retrieval win/loss analysis.
2. Resolution-aware retrieval and incident-to-KB retrieval.
3. Cross-dataset paper analysis of topology-driven retrieval.
