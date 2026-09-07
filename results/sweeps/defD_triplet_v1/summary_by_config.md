# defD_triplet_v1 by configuration

Notes: Definition D triplet loss sweep: triplet_weight, 3 seeds, all else locked at best config

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a_tw0p7_tm0p2 | 3 | 0.432 ± 0.009 | 0.220 ± 0.006 | 0.397 ± 0.007 | 18.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a_tw0p3_tm0p2 | 3 | 0.432 ± 0.004 | 0.213 ± 0.006 | 0.396 ± 0.003 | 20.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a_tw0p5_tm0p2 | 3 | 0.431 ± 0.005 | 0.213 ± 0.002 | 0.395 ± 0.004 | 17.667 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a | 3 | 0.432 ± 0.003 | 0.211 ± 0.008 | 0.396 ± 0.002 | 16.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a_tw1p0_tm0p2 | 3 | 0.419 ± 0.007 | 0.193 ± 0.011 | 0.381 ± 0.005 | 4.667 |

Observed noise floor (largest unseen_ndcg1 std): 0.011
