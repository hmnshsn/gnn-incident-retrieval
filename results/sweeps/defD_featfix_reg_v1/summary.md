# defD_featfix_reg_v1

Notes: Fixed features + ag=1: lr x dropout x wd regularization sweep, 3 seeds
Total runs: 0
Failed runs: 0

| run_name | best_epoch | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s1_knn0a_ag | 15 | 0.464233 | 0.256499 | 0.429766 | 0.207735 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p2_s1_knn0a_ag | 15 | 0.465933 | 0.255349 | 0.430992 | 0.210585 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p3_s42_knn0a_ag | 20 | 0.467960 | 0.253508 | 0.432377 | 0.214452 |
| concat_embed64_drop0p5_embedding_lr0p0001_h256_l2_do0p3_s42_knn0a_ag | 68 | 0.470052 | 0.250058 | 0.433550 | 0.219994 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p3_s1_knn0a_ag | 15 | 0.467822 | 0.249827 | 0.431652 | 0.217995 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p0005_h256_l2_do0p2_s42_knn0a_ag | 59 | 0.458781 | 0.249367 | 0.424035 | 0.209414 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p0005_h256_l2_do0p3_s1_knn0a_ag | 57 | 0.466620 | 0.249137 | 0.430534 | 0.217482 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p001_h256_l2_do0p3_s42_knn0a_ag | 68 | 0.470431 | 0.249137 | 0.433713 | 0.221294 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p3_s42_knn0a_ag | 20 | 0.468398 | 0.249137 | 0.432018 | 0.219260 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p3_s42_knn0a_ag | 20 | 0.466352 | 0.248907 | 0.430273 | 0.217444 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p0005_h256_l2_do0p3_s42_knn0a_ag | 68 | 0.470294 | 0.248677 | 0.433523 | 0.221616 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p2_s1_knn0a_ag | 15 | 0.463364 | 0.248677 | 0.427743 | 0.214687 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p001_h256_l2_do0p2_s42_knn0a_ag | 59 | 0.459213 | 0.248677 | 0.424280 | 0.210535 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p3_s1_knn0a_ag | 15 | 0.467247 | 0.247987 | 0.430867 | 0.219260 |
| concat_embed64_drop0p5_embedding_lr0p0001_h256_l2_do0p2_s1_knn0a_ag | 22 | 0.464802 | 0.247527 | 0.428751 | 0.217275 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p3_s1_knn0a_ag | 15 | 0.466129 | 0.247297 | 0.429820 | 0.218832 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p001_h256_l2_do0p3_s1_knn0a_ag | 57 | 0.466737 | 0.246377 | 0.430175 | 0.220360 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p0005_h256_l2_do0p2_s1_knn0a_ag | 17 | 0.458520 | 0.246377 | 0.423320 | 0.212143 |
| concat_embed64_drop0p5_embedding_lr0p0001_h256_l2_do0p3_s1_knn0a_ag | 57 | 0.464874 | 0.245917 | 0.428544 | 0.218957 |
| concat_embed64_drop0p5_embedding_lr0p0002_h256_l2_do0p2_s1_knn0a_ag | 17 | 0.457108 | 0.245687 | 0.422028 | 0.211421 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p001_h256_l2_do0p3_s1_knn0a_ag | 24 | 0.461697 | 0.245227 | 0.425780 | 0.216470 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p0005_h256_l2_do0p2_s123_knn0a_ag | 40 | 0.464077 | 0.244997 | 0.427726 | 0.219080 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p001_h256_l2_do0p2_s1_knn0a_ag | 17 | 0.459030 | 0.244997 | 0.423517 | 0.214033 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p0005_h256_l2_do0p3_s1_knn0a_ag | 24 | 0.462364 | 0.243846 | 0.426107 | 0.218517 |
| concat_embed64_drop0p5_embedding_lr0p0002_h256_l2_do0p3_s42_knn0a_ag | 28 | 0.465064 | 0.243616 | 0.428321 | 0.221447 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s42_knn0a_ag | 15 | 0.467639 | 0.243156 | 0.430393 | 0.224483 |
| concat_embed64_drop0p5_embedding_lr0p0001_h256_l2_do0p2_s123_knn0a_ag | 40 | 0.462481 | 0.242926 | 0.426052 | 0.219555 |
| concat_embed64_drop0p5_embedding_lr0p0001_h256_l2_do0p2_s42_knn0a_ag | 59 | 0.461547 | 0.241776 | 0.425082 | 0.219771 |
| concat_embed64_drop0p5_embedding_lr0p0002_h256_l2_do0p3_s1_knn0a_ag | 24 | 0.460599 | 0.240626 | 0.424100 | 0.219973 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p2_s42_knn0a_ag | 10 | 0.466665 | 0.240626 | 0.429160 | 0.226040 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p001_h256_l2_do0p3_s123_knn0a_ag | 40 | 0.465207 | 0.240626 | 0.427944 | 0.224582 |
| concat_embed64_drop0p5_embedding_lr0p0002_h256_l2_do0p3_s123_knn0a_ag | 28 | 0.463560 | 0.240166 | 0.426494 | 0.223394 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p2_s42_knn0a_ag | 10 | 0.465292 | 0.239245 | 0.427786 | 0.226047 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p0005_h256_l2_do0p3_s42_knn0a_ag | 28 | 0.466397 | 0.239245 | 0.428708 | 0.227152 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p001_h256_l2_do0p2_s123_knn0a_ag | 40 | 0.462481 | 0.238785 | 0.425365 | 0.223696 |
| concat_embed64_drop0p5_embedding_lr0p0002_h256_l2_do0p2_s42_knn0a_ag | 28 | 0.464979 | 0.238555 | 0.427410 | 0.226423 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p001_h256_l2_do0p2_s123_knn0a_ag | 22 | 0.460507 | 0.238555 | 0.423680 | 0.221952 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p0005_h256_l2_do0p2_s42_knn0a_ag | 28 | 0.464998 | 0.238325 | 0.427388 | 0.226673 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p001_h256_l2_do0p3_s123_knn0a_ag | 28 | 0.460952 | 0.238095 | 0.423975 | 0.222856 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p001_h256_l2_do0p3_s42_knn0a_ag | 28 | 0.464416 | 0.237865 | 0.426827 | 0.226551 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p2_s123_knn0a_ag | 8 | 0.458480 | 0.237865 | 0.421875 | 0.220615 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p3_s123_knn0a_ag | 8 | 0.463756 | 0.237635 | 0.426238 | 0.226121 |
| concat_embed64_drop0p5_embedding_lr0p0001_h256_l2_do0p3_s123_knn0a_ag | 40 | 0.464168 | 0.237175 | 0.426505 | 0.226993 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p0005_h256_l2_do0p3_s123_knn0a_ag | 40 | 0.464443 | 0.235795 | 0.426505 | 0.228648 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p0005_h256_l2_do0p2_s1_knn0a_ag | 28 | 0.462253 | 0.235565 | 0.424640 | 0.226688 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p0005_h256_l2_do0p2_s123_knn0a_ag | 22 | 0.460134 | 0.234875 | 0.422759 | 0.225260 |
| concat_embed64_drop0p5_embedding_lr0p0005_h256_l2_do0p3_s123_knn0a_ag | 10 | 0.460285 | 0.234645 | 0.422846 | 0.225640 |
| concat_embed64_drop0p5_embedding_lr0p0002_h256_l2_do0p2_s123_knn0a_ag | 22 | 0.458840 | 0.233494 | 0.421450 | 0.225346 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p2_s123_knn0a_ag | 8 | 0.458971 | 0.233034 | 0.421483 | 0.225936 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0005_h256_l2_do0p2_s123_knn0a_ag | 8 | 0.458461 | 0.232344 | 0.420943 | 0.226117 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p001_h256_l2_do0p3_s123_knn0a_ag | 10 | 0.461468 | 0.231654 | 0.423337 | 0.229814 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p0005_h256_l2_do0p3_s123_knn0a_ag | 28 | 0.461952 | 0.231194 | 0.423664 | 0.230758 |
| concat_embed64_drop0p5_embedding_lr0p0001_wd0p001_h256_l2_do0p2_s1_knn0a_ag | 26 | 0.459075 | 0.230274 | 0.421112 | 0.228802 |
| concat_embed64_drop0p5_embedding_lr0p0002_wd0p001_h256_l2_do0p2_s42_knn0a_ag | 28 | 0.464704 | 0.227053 | 0.425273 | 0.237651 |
