# concat_seed_variance_v1 by configuration

Notes: Noise floor: 2 configs x 5 seeds at full training length

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p7_embedding_lr0p0005_wd0p0001_h128_l2_do0p3 | 5 | 0.685 ± 0.003 | 0.339 ± 0.007 | 0.627 ± 0.003 | 35.400 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3 | 5 | 0.691 ± 0.004 | 0.335 ± 0.006 | 0.632 ± 0.003 | 31.600 |

Observed noise floor (largest unseen_ndcg1 std): 0.007
