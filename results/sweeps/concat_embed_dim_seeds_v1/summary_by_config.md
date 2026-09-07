# concat_embed_dim_seeds_v1 by configuration

Notes: entity_embed_dim 32/64/128 at concat features, 5 seeds each

| config_label | n_seeds | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | best_epoch_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed32_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3 | 5 | 0.688 ± 0.003 | 0.331 ± 0.011 | 0.628 ± 0.003 | 38.800 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3 | 5 | 0.691 ± 0.003 | 0.330 ± 0.005 | 0.632 ± 0.003 | 30.800 |
| concat_embed128_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3 | 5 | 0.690 ± 0.004 | 0.315 ± 0.010 | 0.628 ± 0.004 | 25.600 |

Observed noise floor (largest unseen_ndcg1 std): 0.011
