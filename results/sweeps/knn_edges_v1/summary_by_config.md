# knn_edges_v1 by configuration

Notes: Text-kNN incident-to-incident edges, k in 0/5/10/25, 5 seeds

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn5a | 5 | 0.688 ± 0.002 | 0.339 ± 0.012 | 0.630 ± 0.003 | 35.400 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 5 | 0.692 ± 0.004 | 0.333 ± 0.007 | 0.632 ± 0.003 | 31.200 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn10a | 5 | 0.685 ± 0.002 | 0.325 ± 0.005 | 0.625 ± 0.001 | 37.600 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn25a | 5 | 0.677 ± 0.006 | 0.322 ± 0.016 | 0.618 ± 0.007 | 38.600 |

Observed noise floor (largest unseen_ndcg1 std): 0.016
