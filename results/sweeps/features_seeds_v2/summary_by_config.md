# features_seeds_v2 by configuration

Notes: Re-run after relevance null-matching fix: categorical vs text vs concat, 5 seeds

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 5 | 0.690 ± 0.002 | 0.140 ± 0.008 | 0.599 ± 0.002 | 30.400 |
| text_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 5 | 0.683 ± 0.003 | 0.136 ± 0.003 | 0.592 ± 0.003 | 31.400 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 5 | 0.658 ± 0.006 | 0.128 ± 0.010 | 0.570 ± 0.005 | 25.600 |

Observed noise floor (largest unseen_ndcg1 std): 0.010
