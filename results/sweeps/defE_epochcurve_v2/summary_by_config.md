# defE_epochcurve_v2 by configuration

Notes: 

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_tw1p0_tm0p3_cltriplet_t0p07_ag | 3 | 0.925 ± 0.005 | 0.782 ± 0.016 | 0.901 ± 0.006 | 7.667 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_ag | 3 | 0.920 ± 0.003 | 0.742 ± 0.010 | 0.890 ± 0.001 | 10.667 |

Observed noise floor (largest unseen_ndcg1 std): 0.016
