# defD_featfix_reg_v2 by configuration

Notes: Fixed features + ag=1: lr x dropout x wd x triplet, 3 seeds

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.465 ± 0.003 | 0.248 ± 0.012 | 0.429 ± 0.004 | 15.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.462 ± 0.003 | 0.248 ± 0.015 | 0.426 ± 0.004 | 11.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.465 ± 0.002 | 0.245 ± 0.007 | 0.429 ± 0.003 | 14.333 |
| concat_embed64_drop0p5_embedding_lr0p0001_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.463 ± 0.002 | 0.244 ± 0.004 | 0.426 ± 0.002 | 40.333 |
| concat_embed64_drop0p5_embedding_lr0p0001_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.465 ± 0.003 | 0.244 ± 0.003 | 0.428 ± 0.003 | 46.333 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p0005_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.465 ± 0.004 | 0.243 ± 0.007 | 0.428 ± 0.004 | 46.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p2_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.460 ± 0.003 | 0.242 ± 0.007 | 0.424 ± 0.003 | 12.000 |
| concat_embed64_drop0p5_embedding_lr0p0002_h256_l2_do0p2_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.458 ± 0.004 | 0.241 ± 0.004 | 0.422 ± 0.003 | 29.000 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p0005_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.462 ± 0.002 | 0.240 ± 0.005 | 0.426 ± 0.002 | 26.667 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p0005_h256_l2_do0p3_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.453 ± 0.003 | 0.240 ± 0.006 | 0.418 ± 0.002 | 41.667 |
| concat_embed64_drop0p5_embedding_lr0p0002_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.463 ± 0.001 | 0.239 ± 0.003 | 0.426 ± 0.001 | 26.667 |
| concat_embed64_drop0p5_embedding_lr0p0002_h256_l2_do0p3_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.454 ± 0.005 | 0.238 ± 0.008 | 0.418 ± 0.005 | 29.333 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p0005_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.460 ± 0.003 | 0.238 ± 0.003 | 0.423 ± 0.002 | 22.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.462 ± 0.004 | 0.238 ± 0.004 | 0.425 ± 0.003 | 9.667 |
| concat_embed64_drop0p5_embedding_lr0p0002_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.460 ± 0.003 | 0.238 ± 0.004 | 0.424 ± 0.002 | 22.333 |
| concat_embed64_drop0p5_embedding_lr0p0001_h256_l2_do0p2_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.459 ± 0.004 | 0.238 ± 0.004 | 0.422 ± 0.003 | 57.333 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p0005_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.460 ± 0.002 | 0.238 ± 0.007 | 0.423 ± 0.003 | 41.667 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p0005_h256_l2_do0p2_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.460 ± 0.003 | 0.237 ± 0.007 | 0.423 ± 0.002 | 30.000 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p0005_h256_l2_do0p2_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.452 ± 0.006 | 0.237 ± 0.006 | 0.416 ± 0.006 | 35.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.462 ± 0.005 | 0.236 ± 0.005 | 0.425 ± 0.004 | 13.667 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p0005_h256_l2_do0p3_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.450 ± 0.003 | 0.234 ± 0.010 | 0.414 ± 0.003 | 26.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p3_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.462 ± 0.006 | 0.232 ± 0.003 | 0.424 ± 0.005 | 19.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p3_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.463 ± 0.009 | 0.231 ± 0.007 | 0.425 ± 0.007 | 14.333 |
| concat_embed64_drop0p5_embedding_lr0p0001_h256_l2_do0p3_s123_knn0a_tw0p7_tm0p2_ag | 3 | 0.453 ± 0.001 | 0.229 ± 0.001 | 0.415 ± 0.001 | 64.333 |

Observed noise floor (largest unseen_ndcg1 std): 0.015
