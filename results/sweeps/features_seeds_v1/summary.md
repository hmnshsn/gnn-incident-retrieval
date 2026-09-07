# features_seeds_v1

Notes: categorical vs text vs concat, 5 seeds each, hyperparameters held fixed
Total runs: 0
Failed runs: 0

| run_name | best_epoch | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1 | 34 | 0.688672 | 0.352065 | 0.632821 | 0.336606 |
| text_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123 | 39 | 0.688933 | 0.335601 | 0.630308 | 0.353332 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3 | 31 | 0.687600 | 0.335305 | 0.629212 | 0.352295 |
| text_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1 | 30 | 0.684422 | 0.334221 | 0.626316 | 0.350202 |
| text_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3 | 35 | 0.684677 | 0.332939 | 0.626251 | 0.351738 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123 | 40 | 0.693451 | 0.332840 | 0.633617 | 0.360610 |
| text_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2 | 26 | 0.683357 | 0.330376 | 0.624724 | 0.352981 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2 | 32 | 0.695504 | 0.322488 | 0.633726 | 0.373015 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42 | 25 | 0.688247 | 0.320319 | 0.627134 | 0.367927 |
| text_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42 | 32 | 0.684207 | 0.320122 | 0.623797 | 0.364084 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1 | 12 | 0.657011 | 0.301094 | 0.599183 | 0.355916 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123 | 32 | 0.656632 | 0.296855 | 0.597853 | 0.359777 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2 | 26 | 0.662025 | 0.284728 | 0.600099 | 0.377297 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3 | 23 | 0.656422 | 0.283052 | 0.594276 | 0.373370 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42 | 43 | 0.654834 | 0.265602 | 0.591446 | 0.389232 |
