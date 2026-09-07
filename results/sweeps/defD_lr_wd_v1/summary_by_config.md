# defD_lr_wd_v1 by configuration

Notes: Definition D Phase 2: lr x weight_decay at best arch h256/do0.1, 5 seeds

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p001_h256_l2_do0p1_s123_knn0a | 5 | 0.431 ± 0.003 | 0.220 ± 0.014 | 0.396 ± 0.003 | 10.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p1_s123_knn0a | 5 | 0.434 ± 0.004 | 0.216 ± 0.010 | 0.398 ± 0.002 | 17.400 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p0005_h256_l2_do0p1_s123_knn0a | 5 | 0.432 ± 0.008 | 0.214 ± 0.007 | 0.396 ± 0.006 | 26.600 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p001_h256_l2_do0p1_s123_knn0a | 5 | 0.431 ± 0.007 | 0.214 ± 0.008 | 0.395 ± 0.004 | 25.600 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p1_s123_knn0a | 5 | 0.432 ± 0.003 | 0.214 ± 0.006 | 0.395 ± 0.002 | 17.600 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a | 5 | 0.432 ± 0.003 | 0.213 ± 0.007 | 0.396 ± 0.002 | 17.400 |
| concat_embed64_drop0p5_embedding_lr0p0003_h256_l2_do0p1_s123_knn0a | 5 | 0.432 ± 0.009 | 0.213 ± 0.008 | 0.395 ± 0.007 | 26.600 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p001_h256_l2_do0p1_s123_knn0a | 5 | 0.433 ± 0.003 | 0.211 ± 0.004 | 0.396 ± 0.003 | 9.800 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p0005_h256_l2_do0p1_s123_knn0a | 5 | 0.432 ± 0.001 | 0.210 ± 0.009 | 0.395 ± 0.001 | 10.000 |

Observed noise floor (largest unseen_ndcg1 std): 0.014
