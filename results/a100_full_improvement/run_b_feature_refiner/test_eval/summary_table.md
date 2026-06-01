| method | n | chamfer | cd_pred_to_gt | cd_gt_to_pred | fscore | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchor | 1200 | 0.042830 | 0.019913 | 0.022917 | 0.596364 | 0.625534 | 0.573579 |
| coarse | 1200 | 0.043141 | 0.019714 | 0.023427 | 0.586850 | 0.627339 | 0.554420 |
| partial | 1200 | 0.151631 | 0.018025 | 0.133606 | 0.326790 | 0.670375 | 0.222509 |
| refined | 1200 | 0.041547 | 0.019784 | 0.021763 | 0.613169 | 0.627135 | 0.603693 |

| comparison | CD improvement vs coarse |
|---|---:|
| anchor_vs_coarse_cd_percent | 0.72% |
| partial_vs_coarse_cd_percent | -251.48% |
| refined_vs_coarse_cd_percent | 3.69% |
