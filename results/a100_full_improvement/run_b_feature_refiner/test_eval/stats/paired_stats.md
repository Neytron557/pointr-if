# Paired Real-Result Statistics

Source: `outputs/a100_full_improvement/run_b_feature_refiner/test_eval/per_sample_metrics.csv`

## Aggregate Methods

| method | n | chamfer | fscore |
|---|---:|---:|---:|
| anchor | 1200 | 0.042830 | 0.596364 |
| coarse | 1200 | 0.043141 | 0.586850 |
| partial | 1200 | 0.151631 | 0.326790 |
| refined | 1200 | 0.041547 | 0.613169 |

## Paired Delta

Baseline: `coarse`
Candidate: `refined`

| metric | value |
|---|---:|
| paired samples | 1200 |
| baseline Chamfer | 0.043141 |
| candidate Chamfer | 0.041547 |
| mean CD delta | 0.00159346 |
| median CD delta | 0.00155813 |
| mean CD improvement | 3.6936% |
| bootstrap 95% CI | [3.6073%, 3.7810%] |
| positive / negative / zero samples | 1186 / 14 / 0 |
| positive fraction | 0.9883 |
| mean F-score delta | 0.02631900 |
| paired t-test p-value | 0.0 |
| Wilcoxon p-value | 2.737593353866877e-194 |

## Oracle Upper Bound

This row is non-deployable because it chooses the better of baseline and candidate using ground truth per sample.

| label | n | chamfer | fscore | candidate chosen | baseline chosen |
|---|---:|---:|---:|---:|---:|
| Oracle | 1200 | 0.041539 | 0.613007 | 1186 | 14 |

## Per-Category Improvement

| category | n | baseline CD | candidate CD | mean CD delta | improvement | positive | negative | zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 02691156 | 150 | 0.029054 | 0.027966 | 0.00108773 | 3.7438% | 149 | 1 | 0 |
| 02933112 | 150 | 0.055852 | 0.053793 | 0.00205938 | 3.6872% | 146 | 4 | 0 |
| 02958343 | 150 | 0.052337 | 0.050392 | 0.00194489 | 3.7161% | 150 | 0 | 0 |
| 03001627 | 150 | 0.045148 | 0.043532 | 0.00161610 | 3.5795% | 149 | 1 | 0 |
| 03636649 | 150 | 0.033420 | 0.032252 | 0.00116818 | 3.4954% | 146 | 4 | 0 |
| 04256520 | 150 | 0.054430 | 0.052374 | 0.00205546 | 3.7763% | 148 | 2 | 0 |
| 04379243 | 150 | 0.039263 | 0.037636 | 0.00162627 | 4.1420% | 149 | 1 | 0 |
| 04530566 | 150 | 0.035623 | 0.034433 | 0.00118968 | 3.3396% | 149 | 1 | 0 |
