# concat_seed_variance_v2 by configuration

Notes: Re-run after relevance fix: noise floor, 2 configs x 5 seeds

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p7_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 5 | 0.685 ± 0.003 | 0.137 ± 0.008 | 0.594 ± 0.002 | 40.800 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 5 | 0.690 ± 0.001 | 0.136 ± 0.003 | 0.598 ± 0.001 | 34.200 |

Observed noise floor (largest unseen_ndcg1 std): 0.008
