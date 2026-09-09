# defD_ag_v1 by configuration

Notes: Definition D assignment_group ablation: ag x triplet_weight, 3 seeds

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.445 ± 0.005 | 0.240 ± 0.005 | 0.411 ± 0.004 | 11.667 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a_ag | 3 | 0.441 ± 0.007 | 0.240 ± 0.008 | 0.408 ± 0.006 | 8.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a | 3 | 0.433 ± 0.003 | 0.215 ± 0.005 | 0.396 ± 0.002 | 16.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a_tw0p7_tm0p2 | 3 | 0.423 ± 0.003 | 0.210 ± 0.005 | 0.388 ± 0.002 | 14.667 |

Observed noise floor (largest unseen_ndcg1 std): 0.008
