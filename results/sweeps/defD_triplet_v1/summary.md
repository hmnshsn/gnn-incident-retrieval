# defD_triplet_v1

Notes: Definition D triplet loss sweep: triplet_weight, 3 seeds, all else locked at best config
Total runs: 0
Failed runs: 0

| run_name | best_epoch | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a_tw0p7_tm0p2 | 12 | 0.420020 | 0.225903 | 0.387583 | 0.194117 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a | 16 | 0.430121 | 0.221762 | 0.395549 | 0.208359 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s42_knn0a_tw0p7_tm0p2 | 22 | 0.439260 | 0.220382 | 0.403058 | 0.218878 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s1_knn0a_tw0p3_tm0p2 | 24 | 0.428454 | 0.218772 | 0.393663 | 0.209682 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s42_knn0a_tw0p3_tm0p2 | 22 | 0.437044 | 0.215321 | 0.400141 | 0.221723 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s42_knn0a_tw0p5_tm0p2 | 10 | 0.424610 | 0.215321 | 0.389770 | 0.209289 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a_tw0p5_tm0p2 | 19 | 0.432161 | 0.212560 | 0.395724 | 0.219600 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s1_knn0a_tw0p7_tm0p2 | 20 | 0.437593 | 0.212330 | 0.400103 | 0.225263 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s1_knn0a_tw0p5_tm0p2 | 24 | 0.437691 | 0.209800 | 0.399764 | 0.227891 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s1_knn0a | 18 | 0.430474 | 0.207039 | 0.393287 | 0.223434 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a_tw0p3_tm0p2 | 14 | 0.431709 | 0.205199 | 0.394126 | 0.226510 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s42_knn0a | 15 | 0.436809 | 0.204739 | 0.398303 | 0.232070 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s42_knn0a_tw1p0_tm0p2 | 7 | 0.415823 | 0.204739 | 0.380571 | 0.211084 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a_tw1p0_tm0p2 | 3 | 0.411234 | 0.196687 | 0.375750 | 0.214547 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s1_knn0a_tw1p0_tm0p2 | 4 | 0.428748 | 0.178054 | 0.387381 | 0.250694 |
