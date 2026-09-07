# features_seeds_v1 by configuration

Notes: categorical vs text vs concat, 5 seeds each, hyperparameters held fixed

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3 | 5 | 0.691 ± 0.003 | 0.333 ± 0.011 | 0.631 ± 0.003 | 32.400 |
| text_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3 | 5 | 0.685 ± 0.002 | 0.331 ± 0.006 | 0.626 ± 0.002 | 32.400 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3 | 5 | 0.657 ± 0.002 | 0.286 ± 0.012 | 0.597 ± 0.003 | 27.200 |

Observed noise floor (largest unseen_ndcg1 std): 0.012
