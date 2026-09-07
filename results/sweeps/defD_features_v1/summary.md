# defD_features_v1

Notes: Definition D adopted: concat vs categorical at best config, 5 seeds
Total runs: 0
Failed runs: 0

| run_name | best_epoch | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1_knn0a | 29 | 0.435436 | 0.220152 | 0.399715 | 0.215284 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3_knn0a | 31 | 0.431951 | 0.209570 | 0.395053 | 0.222381 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42_knn0a | 32 | 0.424786 | 0.205889 | 0.388466 | 0.218897 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2_knn0a | 32 | 0.437456 | 0.205429 | 0.398957 | 0.232027 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 32 | 0.362504 | 0.199908 | 0.335525 | 0.162596 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 28 | 0.432435 | 0.198528 | 0.393625 | 0.233907 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1_knn0a | 12 | 0.377030 | 0.192086 | 0.346344 | 0.184943 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3_knn0a | 23 | 0.371519 | 0.187946 | 0.341060 | 0.183573 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42_knn0a | 43 | 0.380482 | 0.178514 | 0.346971 | 0.201968 |
| categorical_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2_knn0a | 26 | 0.391739 | 0.174833 | 0.355750 | 0.216906 |
