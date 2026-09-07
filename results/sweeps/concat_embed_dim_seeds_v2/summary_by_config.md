# concat_embed_dim_seeds_v2 by configuration

Notes: Re-run after relevance fix: entity_embed_dim 32/64/128 at concat, 5 seeds

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 5 | 0.690 ± 0.003 | 0.136 ± 0.003 | 0.598 ± 0.003 | 31.800 |
| concat_embed128_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 5 | 0.690 ± 0.004 | 0.135 ± 0.005 | 0.598 ± 0.004 | 24.800 |
| concat_embed32_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 5 | 0.684 ± 0.004 | 0.134 ± 0.008 | 0.593 ± 0.004 | 37.000 |

Observed noise floor (largest unseen_ndcg1 std): 0.008
