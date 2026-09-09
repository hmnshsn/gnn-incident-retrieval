# defD_ag_reg_v1 by configuration

Notes: Definition D ag=1 regularization: dropout sweep, 3 seeds

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.447 ± 0.004 | 0.245 ± 0.004 | 0.413 ± 0.003 | 11.667 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.446 ± 0.004 | 0.244 ± 0.002 | 0.412 ± 0.003 | 11.667 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p1_s123_knn0a_ag | 3 | 0.446 ± 0.001 | 0.244 ± 0.006 | 0.412 ± 0.000 | 11.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a_ag | 3 | 0.447 ± 0.004 | 0.242 ± 0.008 | 0.413 ± 0.002 | 8.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.448 ± 0.004 | 0.240 ± 0.006 | 0.414 ± 0.004 | 15.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p3_s123_knn0a_ag | 3 | 0.449 ± 0.006 | 0.239 ± 0.006 | 0.414 ± 0.006 | 15.000 |

Observed noise floor (largest unseen_ndcg1 std): 0.008
