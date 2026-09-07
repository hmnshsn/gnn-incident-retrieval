# concat_embed_dim_seeds_v2

Notes: Re-run after relevance fix: entity_embed_dim 32/64/128 at concat, 5 seeds
Total runs: 0
Failed runs: 0

| run_name | best_epoch | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed128_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 30 | 0.689404 | 0.142857 | 0.598720 | 0.546547 |
| concat_embed32_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3_knn0a | 36 | 0.678754 | 0.142857 | 0.589837 | 0.535897 |
| concat_embed32_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1_knn0a | 45 | 0.691287 | 0.141477 | 0.600061 | 0.549810 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42_knn0a | 39 | 0.685776 | 0.138716 | 0.595006 | 0.547059 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2_knn0a | 32 | 0.694876 | 0.138026 | 0.602482 | 0.556850 |
| concat_embed128_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2_knn0a | 26 | 0.690574 | 0.137336 | 0.598780 | 0.553238 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 28 | 0.689424 | 0.136646 | 0.597705 | 0.552778 |
| concat_embed128_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3_knn0a | 23 | 0.687436 | 0.135266 | 0.595819 | 0.552171 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3_knn0a | 31 | 0.689607 | 0.135266 | 0.597629 | 0.554341 |
| concat_embed32_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 39 | 0.684128 | 0.133195 | 0.592716 | 0.550933 |
| concat_embed32_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2_knn0a | 37 | 0.683357 | 0.131815 | 0.591844 | 0.551542 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1_knn0a | 29 | 0.688070 | 0.130435 | 0.595546 | 0.557636 |
| concat_embed128_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1_knn0a | 19 | 0.685214 | 0.129745 | 0.593049 | 0.555469 |
| concat_embed128_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42_knn0a | 26 | 0.698079 | 0.129745 | 0.603780 | 0.568335 |
| concat_embed32_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42_knn0a | 28 | 0.682755 | 0.119393 | 0.589281 | 0.563363 |
