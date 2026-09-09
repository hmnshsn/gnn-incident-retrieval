# defE_margin_v1 by configuration

Notes: 

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw1p0_tm0p3_ag | 3 | 0.919 ± 0.001 | 0.795 ± 0.003 | 0.898 ± 0.002 | 5.000 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw1p0_tm1p5_ag | 3 | 0.924 ± 0.004 | 0.795 ± 0.003 | 0.903 ± 0.003 | 2.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw1p0_tm0p7_ag | 3 | 0.923 ± 0.003 | 0.794 ± 0.003 | 0.901 ± 0.003 | 3.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw1p0_tm0p5_ag | 3 | 0.924 ± 0.004 | 0.794 ± 0.002 | 0.902 ± 0.004 | 2.333 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw1p0_tm1_ag | 3 | 0.923 ± 0.003 | 0.793 ± 0.003 | 0.902 ± 0.003 | 3.333 |

Observed noise floor (largest unseen_ndcg1 std): 0.003
