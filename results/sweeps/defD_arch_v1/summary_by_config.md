# defD_arch_v1 by configuration

Notes: Definition D architecture sweep: hidden_dim x dropout, 3 seeds

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a | 3 | 0.432 ± 0.002 | 0.219 ± 0.004 | 0.396 ± 0.001 | 18.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p5_s123_knn0a | 3 | 0.432 ± 0.003 | 0.216 ± 0.011 | 0.396 ± 0.004 | 29.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p3_s123_knn0a | 3 | 0.431 ± 0.007 | 0.214 ± 0.003 | 0.395 ± 0.006 | 21.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h128_l2_do0p1_s123_knn0a | 3 | 0.429 ± 0.002 | 0.207 ± 0.003 | 0.392 ± 0.002 | 24.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h128_l2_do0p3_s123_knn0a | 3 | 0.432 ± 0.010 | 0.203 ± 0.005 | 0.394 ± 0.008 | 31.667 |
| concat_embed64_drop0p5_embedding_lr0p0005_h64_l2_do0p1_s123_knn0a | 3 | 0.429 ± 0.004 | 0.202 ± 0.013 | 0.391 ± 0.004 | 29.667 |
| concat_embed64_drop0p5_embedding_lr0p0005_h128_l2_do0p5_s123_knn0a | 3 | 0.429 ± 0.004 | 0.198 ± 0.002 | 0.390 ± 0.004 | 37.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h64_l2_do0p3_s123_knn0a | 3 | 0.427 ± 0.002 | 0.188 ± 0.006 | 0.388 ± 0.001 | 41.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h64_l2_do0p5_s123_knn0a | 3 | 0.424 ± 0.004 | 0.179 ± 0.006 | 0.383 ± 0.002 | 49.667 |

Observed noise floor (largest unseen_ndcg1 std): 0.013
