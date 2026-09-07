# concat_embed_dim_seeds_v1

Notes: entity_embed_dim 32/64/128 at concat features, 5 seeds each
Total runs: 0
Failed runs: 0

| run_name | best_epoch | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed32_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1 | 45 | 0.689182 | 0.349699 | 0.632789 | 0.339482 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2 | 32 | 0.694791 | 0.335699 | 0.635668 | 0.359091 |
| concat_embed128_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2 | 26 | 0.689404 | 0.333925 | 0.630357 | 0.355479 |
| concat_embed32_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123 | 40 | 0.683684 | 0.333925 | 0.625585 | 0.349759 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1 | 29 | 0.689796 | 0.332150 | 0.630455 | 0.357646 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123 | 28 | 0.694268 | 0.331854 | 0.634135 | 0.362413 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42 | 34 | 0.688430 | 0.328897 | 0.628775 | 0.359533 |
| concat_embed32_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2 | 33 | 0.692163 | 0.328404 | 0.631873 | 0.363759 |
| concat_embed32_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3 | 36 | 0.683886 | 0.326333 | 0.624560 | 0.357553 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3 | 31 | 0.689907 | 0.321798 | 0.628764 | 0.368109 |
| concat_embed32_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42 | 40 | 0.688672 | 0.317657 | 0.627112 | 0.371014 |
| concat_embed128_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3 | 27 | 0.689384 | 0.311939 | 0.626758 | 0.377445 |
| concat_embed128_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123 | 30 | 0.691385 | 0.310362 | 0.628165 | 0.381023 |
| concat_embed128_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42 | 26 | 0.697327 | 0.308883 | 0.632876 | 0.388445 |
| concat_embed128_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1 | 19 | 0.684887 | 0.308094 | 0.622303 | 0.376792 |
