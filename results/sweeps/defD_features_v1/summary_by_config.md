# defD_features_v1 by configuration

Notes: Definition D adopted: concat vs categorical at best config, 5 seeds

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 5 | 0.432 ± 0.004 | 0.208 ± 0.007 | 0.395 ± 0.004 | 30.400 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 5 | 0.377 ± 0.010 | 0.187 ± 0.009 | 0.345 ± 0.007 | 27.200 |

Observed noise floor (largest unseen_ndcg1 std): 0.009
