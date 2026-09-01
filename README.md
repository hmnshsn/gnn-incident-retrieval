# GNN Incident Retrieval

Graph Neural Networks for incident-to-incident retrieval in IT Service Management.

## Datasets

| Dataset             | Incidents | CIs   | Resolution Codes | Categories | Subcategories  | Text Fields                    |
| ------------------- | --------- | ----- | ---------------- | ---------- | -------------- | ------------------------------ |
| BPI 2014 (Rabobank) | 46,146    | 2,794 | 14               | 4          | —              | None                           |
| ServiceNow Internal | 59,151    | 7,140 | 25               | 36         | 154 (57% null) | short_description, description |

## Model Architecture

**GNN: Heterogeneous GraphSAGE (HeteroSAGE)**

2-layer GraphSAGE convolutions wrapped in PyTorch Geometric's HeteroConv with mean aggregation across edge types. Each edge type (incident-to-CI, incident-to-category, CI-to-subtype, etc.) has its own SAGEConv layer. Non-incident nodes (CI, category, subtype) use learnable nn.Embedding tables. Incident nodes use TF-IDF features over available text fields. For closure-code prediction, a linear classification head produces logits. For incident retrieval, embeddings are extracted from the pre-classifier hidden layer and ranked by cosine similarity.

**Graph structure:**
- Node types: incident, ci, ci_subtype, category (BPI 2014 also includes ci_type and ci_cby)
- Edge types: bidirectional edges between incidents and their CI, category, and subtype nodes
- No target leakage: closure codes are classification targets only, not graph nodes

## File Structure

- `scripts/09_encode_sn_text.py` — precompute sentence embeddings for SN incidents
- `scripts/10_extract_sn_gnn_embeddings.py` — extract and cache GNN embeddings + split indices
- `scripts/11_run_sn_text_retrieval.py` — text-only, GNN-only, and late fusion retrieval evaluation
- `scripts/12_run_sn_cold_start_analysis.py` — SN seen vs unseen CI retrieval analysis
- `scripts/13_run_sn_early_fusion.py` — early fusion ablation (tfidf/text/concat features)
- `scripts/14_run_sn_adaptive_alpha.py` — adaptive alpha by CI visibility (negative result)
- `scripts/15_run_bpi_cold_start_analysis.py` — BPI 2014 cold-start replication
- `scripts/alpha_sweep.sh` — alpha sweep over late fusion weights
- `src/text_encoder.py` — sentence embedding encoding with caching

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

## Exp02: Closure Code Prediction (MRR)

| Dataset | GNN | CI-Majority | Delta |
|---------|-----|-------------|-------|
| BPI 2014 | 0.702 | 0.686 | +0.016 |
| SN | 0.614 | 0.578 | +0.036 |

## Exp03: Incident Retrieval (nDCG@1)

| Dataset | GNN | CI-Subtype | Delta |
|---------|-----|------------|-------|
| BPI 2014 | 0.643 | 0.570 | +0.073 |
| SN (raw subtype) | 0.596 | 0.579 | +0.017 |

## Subtype Ablation (SN, GNN only)

| Mode | nDCG@1 | MAP | Notes |
|------|--------|-----|-------|
| none | 0.572 | 0.373 | No subtype edges |
| raw | 0.596 | 0.380 | Best top-1, preferred for ZTSD |
| imputed | 0.583 | 0.425 | Best recall |

## Exp04: Text Signal — Late Fusion (SN)

Late fusion combines precomputed sentence embeddings (all-MiniLM-L6-v2, 384-dim) with GNN embeddings post-training. Formula: fused = alpha * norm(gnn) concat (1-alpha) * norm(text). GNN trained with early stopping (best epoch 2, patience 10).

Alpha sweep:

| Alpha | nDCG@1 | nDCG@5 | nDCG@10 | MAP | MRR |
|-------|--------|--------|---------|-----|-----|
| Text-only (0.0) | 0.357 | 0.348 | 0.343 | 0.175 | 0.638 |
| 0.5 | 0.568 | 0.550 | 0.543 | 0.344 | 0.845 |
| 0.6 | 0.606 | 0.592 | 0.587 | 0.375 | 0.873 |
| 0.7 | 0.627 | 0.614 | 0.608 | 0.383 | 0.884 |
| **0.8** | **0.631** | **0.615** | **0.608** | **0.383** | **0.885** |
| 0.85 | 0.629 | 0.614 | 0.606 | 0.382 | 0.883 |
| 0.9 | 0.629 | 0.612 | 0.604 | 0.381 | 0.882 |
| 0.95 | 0.629 | 0.611 | 0.603 | 0.380 | 0.881 |
| GNN-only (1.0) | 0.591 | 0.585 | 0.580 | 0.380 | 0.880 |

Best alpha: 0.8 (80% GNN, 20% text). Text acts as a tiebreaker for top-1 retrieval.

## Exp05: Cold-Start Analysis (SN)

Test set split by CI visibility in training: 7284 seen, 1449 unseen (16.6%).

| Group | Method | nDCG@1 | MAP | MRR |
|-------|--------|--------|-----|-----|
| Seen | Text-only | 0.393 | 0.193 | 0.705 |
| Seen | GNN-only | 0.651 | 0.404 | 0.964 |
| Seen | Late fusion (0.8) | **0.691** | **0.407** | **0.967** |
| Unseen | Text-only | 0.174 | 0.086 | 0.301 |
| Unseen | GNN-only | 0.274 | 0.259 | 0.458 |
| Unseen | Late fusion (0.8) | **0.327** | **0.262** | **0.475** |

Cold-start degradation: GNN nDCG@1 drops 58% (0.651 to 0.274) for unseen CIs. Fusion lift is larger for unseen (+0.053) than seen (+0.040).

## Exp06: Early Fusion Ablation (SN)

Text embeddings as GNN input features instead of post-hoc fusion. All runs use early stopping (patience 10).

| Features | All nDCG@1 | Seen nDCG@1 | Unseen nDCG@1 | All MAP | All MRR |
|----------|-----------|-------------|---------------|---------|---------|
| TF-IDF (baseline) | 0.588 | 0.650 | 0.276 | 0.380 | 0.880 |
| Sentence embeddings | 0.626 | 0.686 | 0.326 | 0.387 | 0.886 |
| TF-IDF + sentence (concat) | 0.624 | 0.681 | **0.336** | **0.389** | **0.888** |
| Late fusion (0.8) | **0.631** | **0.691** | 0.327 | 0.383 | 0.885 |

Late fusion wins nDCG@1. Concat wins MAP and cold-start nDCG@1. Early fusion does not outperform late fusion on the headline metric.

## Exp07: Adaptive Alpha (SN) — Negative Result

Tested CI-aware alpha (alpha_seen=0.9, alpha_unseen=0.6) vs fixed alpha=0.8. No improvement. Cold-start degradation stems from signal quality, not signal weighting.

## Exp08: BPI 2014 Cold-Start Replication

| Group | nDCG@1 | MAP | MRR |
|-------|--------|-----|-----|
| Seen | 0.667 | 0.463 | 0.983 |
| Unseen | 0.339 | 0.163 | 0.411 |
| All | 0.646 | 0.444 | 0.946 |

Cold-start degradation consistent across datasets: nDCG@1 drops 49% on BPI 2014 vs 58% on SN.

## Cross-Dataset Cold-Start Summary

| | BPI 2014 | ServiceNow |
|---|---|---|
| Seen nDCG@1 | 0.667 | 0.651 |
| Unseen nDCG@1 | 0.339 | 0.274 |
| Drop | -49% | -58% |
| Unseen % of test | ~6% | 16.6% |
| CI vocabulary | 2794 | 7140 |

### Key Findings

1. CI identity dominates resolution prediction across both datasets
2. GNN adds value via subtype propagation, especially for cold-start scenarios
3. Late fusion (alpha=0.8) provides consistent nDCG@1 improvement (+0.040 overall) with minimal complexity
4. Cold-start degradation is the primary challenge: 49-58% nDCG@1 drop for unseen CIs across both datasets
5. Text signal helps as a tiebreaker but is too weak standalone to solve cold-start
6. Early fusion matches late fusion; concat features win on MAP and cold-start nDCG@1
7. Adaptive per-group alpha shows no improvement over fixed alpha (negative result)
8. Early stopping finds optimal model at epoch 2-3, consistent across all feature configurations

## What's next

1. Per-incident win/loss analysis — quantify what fraction of test incidents improve vs degrade under fusion
2. Incident-to-KB retrieval — pending KB number field fix from data provider (current export has empty KB number column)
3. Paper writing — cross-dataset cold-start narrative as core contribution
