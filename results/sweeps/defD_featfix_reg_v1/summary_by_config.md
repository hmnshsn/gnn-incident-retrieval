# defD_featfix_reg_v1 by configuration

Notes: Fixed features + ag=1: lr x dropout x wd regularization sweep, 3 seeds

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.467 ± 0.002 | 0.247 ± 0.007 | 0.430 ± 0.003 | 14.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.463 ± 0.004 | 0.246 ± 0.008 | 0.427 ± 0.004 | 12.667 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p001_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.467 ± 0.002 | 0.245 ± 0.004 | 0.431 ± 0.002 | 55.000 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p0005_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.467 ± 0.002 | 0.245 ± 0.006 | 0.430 ± 0.003 | 55.000 |
| concat_embed64_drop0p5_embedding_lr0p0001_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.466 ± 0.003 | 0.244 ± 0.005 | 0.430 ± 0.003 | 55.000 |
| concat_embed64_drop0p5_embedding_lr0p0001_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.463 ± 0.001 | 0.244 ± 0.002 | 0.427 ± 0.002 | 40.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.465 ± 0.003 | 0.244 ± 0.006 | 0.428 ± 0.004 | 15.000 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p0005_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.462 ± 0.002 | 0.243 ± 0.006 | 0.425 ± 0.002 | 42.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.465 ± 0.003 | 0.243 ± 0.008 | 0.428 ± 0.003 | 15.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.463 ± 0.003 | 0.243 ± 0.009 | 0.427 ± 0.004 | 11.000 |
| concat_embed64_drop0p5_embedding_lr0p0002_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.463 ± 0.002 | 0.241 ± 0.002 | 0.426 ± 0.002 | 26.667 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.463 ± 0.003 | 0.241 ± 0.007 | 0.426 ± 0.004 | 11.000 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p001_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.462 ± 0.001 | 0.240 ± 0.003 | 0.426 ± 0.001 | 26.667 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p0005_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.461 ± 0.003 | 0.240 ± 0.005 | 0.424 ± 0.002 | 22.333 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p001_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.460 ± 0.002 | 0.239 ± 0.008 | 0.424 ± 0.002 | 41.667 |
| concat_embed64_drop0p5_embedding_lr0p0002_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.460 ± 0.003 | 0.239 ± 0.005 | 0.424 ± 0.003 | 22.333 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p0005_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.464 ± 0.002 | 0.238 ± 0.005 | 0.426 ± 0.002 | 26.667 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p001_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.461 ± 0.002 | 0.237 ± 0.007 | 0.424 ± 0.001 | 22.333 |

Observed noise floor (largest unseen_ndcg1 std): 0.009
