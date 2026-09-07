# knn_edges_v2

Notes: Re-run after relevance fix: text-kNN edges k 0/5/10/25, 5 seeds
Total runs: 0
Failed runs: 0

| run_name | best_epoch | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn5a | 38 | 0.680866 | 0.142857 | 0.591598 | 0.538009 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1_knn0a | 29 | 0.690359 | 0.142167 | 0.599401 | 0.548192 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn0a | 31 | 0.693209 | 0.141477 | 0.601664 | 0.551732 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2_knn5a | 32 | 0.681383 | 0.140097 | 0.591571 | 0.541286 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42_knn10a | 35 | 0.684259 | 0.140097 | 0.593970 | 0.544162 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn10a | 24 | 0.682108 | 0.139406 | 0.592062 | 0.542702 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1_knn5a | 34 | 0.686266 | 0.138716 | 0.595415 | 0.547550 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2_knn0a | 32 | 0.695615 | 0.138716 | 0.603213 | 0.556898 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2_knn25a | 34 | 0.674512 | 0.136646 | 0.585268 | 0.537866 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42_knn0a | 33 | 0.683677 | 0.135266 | 0.592683 | 0.548412 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1_knn10a | 41 | 0.687953 | 0.134576 | 0.596135 | 0.553377 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42_knn5a | 47 | 0.696288 | 0.133195 | 0.602858 | 0.563093 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s2_knn10a | 27 | 0.679055 | 0.130435 | 0.588027 | 0.548620 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3_knn0a | 31 | 0.690274 | 0.129745 | 0.597269 | 0.560529 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s1_knn25a | 42 | 0.674185 | 0.124914 | 0.583048 | 0.549271 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3_knn5a | 33 | 0.687783 | 0.124914 | 0.594390 | 0.562869 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42_knn25a | 45 | 0.681932 | 0.124914 | 0.589510 | 0.557018 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3_knn25a | 39 | 0.683618 | 0.124224 | 0.590802 | 0.559395 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s123_knn25a | 38 | 0.684501 | 0.120773 | 0.590966 | 0.563728 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s3_knn10a | 27 | 0.678709 | 0.115252 | 0.585219 | 0.563457 |
