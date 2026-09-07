# features_seeds_v2

Notes: Re-run after relevance null-matching fix: categorical vs text vs concat, 5 seeds
Total runs: 0
Failed runs: 0

| run_name | best_epoch | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 28 | 0.690797 | 0.147688 | 0.600683 | 0.543108 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1_knn0a | 29 | 0.689744 | 0.144928 | 0.599347 | 0.544816 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 32 | 0.651500 | 0.144237 | 0.567334 | 0.507262 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42_knn0a | 32 | 0.687796 | 0.142857 | 0.597378 | 0.544939 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2_knn0a | 32 | 0.692594 | 0.140097 | 0.600923 | 0.552498 |
| text_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42_knn0a | 34 | 0.680559 | 0.138716 | 0.590655 | 0.541842 |
| text_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 39 | 0.687554 | 0.138026 | 0.596375 | 0.549528 |
| text_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2_knn0a | 26 | 0.683311 | 0.138026 | 0.592836 | 0.545285 |
| text_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1_knn0a | 29 | 0.678637 | 0.133885 | 0.588250 | 0.544751 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1_knn0a | 12 | 0.657488 | 0.131125 | 0.570153 | 0.526363 |
| text_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3_knn0a | 29 | 0.683376 | 0.131125 | 0.591746 | 0.552252 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2_knn0a | 26 | 0.667556 | 0.126984 | 0.577863 | 0.540572 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3_knn0a | 31 | 0.689299 | 0.125604 | 0.595770 | 0.563696 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3_knn0a | 23 | 0.652526 | 0.121463 | 0.564411 | 0.531063 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42_knn0a | 35 | 0.658586 | 0.115252 | 0.568435 | 0.543334 |
