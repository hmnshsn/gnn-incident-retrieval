# knn_edges_v2 by configuration

Notes: Re-run after relevance fix: text-kNN edges k 0/5/10/25, 5 seeds

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 5 | 0.691 ± 0.004 | 0.137 ± 0.005 | 0.599 ± 0.004 | 31.200 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn5a | 5 | 0.687 ± 0.006 | 0.136 ± 0.006 | 0.595 ± 0.004 | 36.800 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn10a | 5 | 0.682 ± 0.003 | 0.132 ± 0.009 | 0.591 ± 0.004 | 30.800 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn25a | 5 | 0.680 ± 0.004 | 0.126 ± 0.005 | 0.588 ± 0.003 | 39.600 |

Observed noise floor (largest unseen_ndcg1 std): 0.009
