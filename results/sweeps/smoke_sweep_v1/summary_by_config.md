# smoke_sweep_v1 by configuration

Notes: Verify native sweep wiring end to end

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3 | 2 | 0.676 ± 0.001 | 0.289 ± 0.002 | 0.612 ± 0.001 | 3.000 |

Observed noise floor (largest unseen_ndcg1 std): 0.002
