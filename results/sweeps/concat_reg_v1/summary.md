# concat_reg_v1

Notes: Re-tune dropout and lr on top of concat features (input_dim 768)
Total runs: 9
Failed runs: 0

| run_name | best_epoch | seen_ndcg1 | unseen_ndcg1 | all_ndcg1 | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat_embed64_drop0p7_embedding_lr0p0003_wd0p0001_h128_l2_do0p3_s42 | 57 | 0.687913 | 0.334122 | 0.629212 | 0.353791 |
| concat_embed64_drop0p7_embedding_lr0p001_wd0p0001_h128_l2_do0p3_s42 | 25 | 0.690385 | 0.331657 | 0.630929 | 0.358727 |
| concat_embed64_drop0p7_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42 | 39 | 0.687312 | 0.329094 | 0.627876 | 0.358218 |
| concat_embed64_drop0p3_embedding_lr0p001_wd0p0001_h128_l2_do0p3_s42 | 13 | 0.690365 | 0.328502 | 0.630324 | 0.361863 |
| concat_embed64_drop0p5_embedding_lr0p0003_wd0p0001_h128_l2_do0p3_s42 | 47 | 0.690234 | 0.323967 | 0.629462 | 0.366267 |
| concat_embed64_drop0p3_embedding_lr0p0003_wd0p0001_h128_l2_do0p3_s42 | 48 | 0.693196 | 0.323770 | 0.631900 | 0.369426 |
| concat_embed64_drop0p3_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42 | 23 | 0.689966 | 0.321010 | 0.628748 | 0.368957 |
| concat_embed64_drop0p5_embedding_lr0p0005_wd0p0001_h128_l2_do0p3_s42 | 23 | 0.689724 | 0.319136 | 0.628236 | 0.370588 |
| concat_embed64_drop0p5_embedding_lr0p001_wd0p0001_h128_l2_do0p3_s42 | 23 | 0.694726 | 0.310559 | 0.630984 | 0.384167 |
