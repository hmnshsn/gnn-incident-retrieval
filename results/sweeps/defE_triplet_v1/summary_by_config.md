# defE_triplet_v1 by configuration

Notes: 

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw1p0_tm0p3_ag | 3 | 0.924 ± 0.006 | 0.791 ± 0.005 | 0.902 ± 0.006 | 5.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw0p5_tm0p3_ag | 3 | 0.919 ± 0.002 | 0.745 ± 0.005 | 0.890 ± 0.001 | 9.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw0p3_tm0p3_ag | 3 | 0.920 ± 0.001 | 0.745 ± 0.006 | 0.891 ± 0.000 | 9.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.921 ± 0.001 | 0.744 ± 0.009 | 0.892 ± 0.002 | 9.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw0p7_tm0p3_ag | 3 | 0.920 ± 0.002 | 0.742 ± 0.002 | 0.890 ± 0.002 | 9.667 |

Observed noise floor (largest unseen_ndcg1 std): 0.009
