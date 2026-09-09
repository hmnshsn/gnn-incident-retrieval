# defE_infonce_v1 by configuration

Notes: 

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw1p0_tm0p3_clinfonce_t0p07_ag | 3 | 0.936 ± 0.003 | 0.682 ± 0.011 | 0.894 ± 0.004 | 15.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw1p0_tm0p3_clinfonce_t0p05_ag | 3 | 0.933 ± 0.006 | 0.662 ± 0.028 | 0.888 ± 0.008 | 21.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw1p0_tm0p3_clinfonce_t0p1_ag | 3 | 0.922 ± 0.002 | 0.610 ± 0.010 | 0.871 ± 0.003 | 24.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw1p0_tm0p3_clinfonce_t0p2_ag | 3 | 0.856 ± 0.016 | 0.508 ± 0.008 | 0.798 ± 0.014 | 19.000 |

Observed noise floor (largest unseen_ndcg1 std): 0.028
