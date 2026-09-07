# defD_lr_wd_v1

Notes: Definition D Phase 2: lr x weight_decay at best arch h256/do0.1, 5 seeds
Total runs: 0
Failed runs: 0

| run_name | best_epoch | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p001_h256_l2_do0p1_s123_knn0a | 11 | 0.426787 | 0.242466 | 0.396204 | 0.184321 |
| concat_embed64_drop0p5_embedding_lr0p001_h256_l2_do0p1_s42_knn0a | 10 | 0.435161 | 0.230274 | 0.401166 | 0.204887 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p1_s123_knn0a | 16 | 0.431533 | 0.227973 | 0.397758 | 0.203560 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p1_s1_knn0a | 24 | 0.428166 | 0.227283 | 0.394721 | 0.200883 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p0005_h256_l2_do0p1_s123_knn0a | 16 | 0.421040 | 0.226133 | 0.388815 | 0.194907 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s1_knn0a | 24 | 0.428297 | 0.225213 | 0.394601 | 0.203084 |
| concat_embed64_drop0p5_embedding_lr0p0003_h256_l2_do0p1_s1_knn0a | 24 | 0.426493 | 0.224523 | 0.392867 | 0.201970 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p0005_h256_l2_do0p1_s123_knn0a | 11 | 0.429709 | 0.223142 | 0.395549 | 0.206567 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p1_s1_knn0a | 24 | 0.429003 | 0.222452 | 0.394732 | 0.206551 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p001_h256_l2_do0p1_s123_knn0a | 16 | 0.422805 | 0.221302 | 0.389486 | 0.201503 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p001_h256_l2_do0p1_s1_knn0a | 24 | 0.427022 | 0.220612 | 0.392774 | 0.206410 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p001_h256_l2_do0p1_s3_knn0a | 24 | 0.429101 | 0.219232 | 0.394279 | 0.209869 |
| concat_embed64_drop0p5_embedding_lr0p0003_h256_l2_do0p1_s42_knn0a | 33 | 0.439025 | 0.217851 | 0.402327 | 0.221173 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p1_s123_knn0a | 16 | 0.431454 | 0.216241 | 0.395746 | 0.215213 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p001_h256_l2_do0p1_s1_knn0a | 9 | 0.437207 | 0.216241 | 0.400544 | 0.220966 |
| concat_embed64_drop0p5_embedding_lr0p001_h256_l2_do0p1_s3_knn0a | 13 | 0.430049 | 0.215781 | 0.394497 | 0.214268 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p0005_h256_l2_do0p1_s2_knn0a | 7 | 0.432736 | 0.214861 | 0.396700 | 0.217875 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p0005_h256_l2_do0p1_s3_knn0a | 29 | 0.431827 | 0.214401 | 0.395751 | 0.217426 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s123_knn0a | 16 | 0.430474 | 0.214171 | 0.394584 | 0.216303 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p0005_h256_l2_do0p1_s1_knn0a | 24 | 0.425237 | 0.213711 | 0.390026 | 0.211527 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p1_s2_knn0a | 17 | 0.430696 | 0.213251 | 0.394503 | 0.217446 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p001_h256_l2_do0p1_s123_knn0a | 10 | 0.430317 | 0.213020 | 0.394148 | 0.217296 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p001_h256_l2_do0p1_s2_knn0a | 7 | 0.433913 | 0.213020 | 0.397376 | 0.220892 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s42_knn0a | 15 | 0.434985 | 0.212330 | 0.397927 | 0.222654 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p1_s3_knn0a | 16 | 0.432788 | 0.212100 | 0.396171 | 0.220688 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p0005_h256_l2_do0p1_s42_knn0a | 33 | 0.435750 | 0.211870 | 0.398488 | 0.223879 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p001_h256_l2_do0p1_s42_knn0a | 33 | 0.432572 | 0.211640 | 0.396029 | 0.220932 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s3_knn0a | 16 | 0.429199 | 0.210720 | 0.392948 | 0.218479 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p1_s3_knn0a | 16 | 0.430415 | 0.210720 | 0.393848 | 0.219695 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p0005_h256_l2_do0p1_s3_knn0a | 13 | 0.431049 | 0.210720 | 0.394377 | 0.220329 |
| concat_embed64_drop0p5_embedding_lr0p0003_h256_l2_do0p1_s123_knn0a | 16 | 0.419452 | 0.210260 | 0.384742 | 0.209192 |
| concat_embed64_drop0p5_embedding_lr0p0003_h256_l2_do0p1_s3_knn0a | 29 | 0.430023 | 0.208650 | 0.393177 | 0.221373 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p001_h256_l2_do0p1_s3_knn0a | 13 | 0.430258 | 0.207499 | 0.393297 | 0.222759 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p1_s42_knn0a | 15 | 0.437789 | 0.207499 | 0.399579 | 0.230290 |
| concat_embed64_drop0p5_embedding_lr0p001_h256_l2_do0p1_s2_knn0a | 7 | 0.432415 | 0.207269 | 0.395059 | 0.225146 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p0005_h256_l2_do0p1_s1_knn0a | 9 | 0.432984 | 0.206809 | 0.395457 | 0.226175 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p0005_h256_l2_do0p1_s2_knn0a | 31 | 0.444837 | 0.206119 | 0.405228 | 0.238718 |
| concat_embed64_drop0p5_embedding_lr0p001_h256_l2_do0p1_s1_knn0a | 9 | 0.431788 | 0.206119 | 0.394230 | 0.225669 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p1_s42_knn0a | 15 | 0.436887 | 0.205429 | 0.398483 | 0.231458 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p1_s2_knn0a | 16 | 0.435835 | 0.204969 | 0.397529 | 0.230866 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p001_h256_l2_do0p1_s42_knn0a | 10 | 0.432396 | 0.204739 | 0.394737 | 0.227657 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p1_s2_knn0a | 16 | 0.438659 | 0.203129 | 0.399694 | 0.235530 |
| concat_embed64_drop0p5_embedding_lr0p0003_h256_l2_do0p1_s2_knn0a | 31 | 0.444536 | 0.201288 | 0.404061 | 0.243248 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p001_h256_l2_do0p1_s2_knn0a | 31 | 0.442366 | 0.199218 | 0.402022 | 0.243148 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p0005_h256_l2_do0p1_s42_knn0a | 10 | 0.433141 | 0.195077 | 0.393641 | 0.238064 |
