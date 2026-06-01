# Paired Real-Result Statistics

Source: `outputs/a100_full_improvement/run_a_pcn_mvc/test_eval/per_sample_metrics.csv`

## Aggregate Methods

| method | n | chamfer | fscore |
|---|---:|---:|---:|
| coarse | 1200 | 0.040271 | 0.630746 |
| partial | 1200 | 0.147157 | 0.362022 |
| refined | 1200 | 0.048130 | 0.561478 |

## Paired Delta

Baseline: `coarse`
Candidate: `refined`

| metric | value |
|---|---:|
| paired samples | 1200 |
| baseline Chamfer | 0.040271 |
| candidate Chamfer | 0.048130 |
| mean CD delta | -0.00785891 |
| median CD delta | -0.00702460 |
| mean CD improvement | -19.5151% |
| bootstrap 95% CI | [-20.6899%, -18.3939%] |
| positive / negative / zero samples | 104 / 1096 / 0 |
| positive fraction | 0.0867 |
| mean F-score delta | -0.06926757 |
| paired t-test p-value | 4.227581097418119e-198 |
| Wilcoxon p-value | 1.9912765705125048e-174 |

## Oracle Upper Bound

This row is non-deployable because it chooses the better of baseline and candidate using ground truth per sample.

| label | n | chamfer | fscore | candidate chosen | baseline chosen |
|---|---:|---:|---:|---:|---:|
| Oracle | 1200 | 0.040026 | 0.629576 | 104 | 1096 |

## Per-Category Improvement

| category | n | baseline CD | candidate CD | mean CD delta | improvement | positive | negative | zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 02691156 | 150 | 0.027035 | 0.035645 | -0.00861024 | -31.8488% | 2 | 148 | 0 |
| 02933112 | 150 | 0.052810 | 0.056873 | -0.00406300 | -7.6937% | 41 | 109 | 0 |
| 02958343 | 150 | 0.049074 | 0.056538 | -0.00746342 | -15.2084% | 28 | 122 | 0 |
| 03001627 | 150 | 0.041918 | 0.049855 | -0.00793770 | -18.9364% | 8 | 142 | 0 |
| 03636649 | 150 | 0.031580 | 0.040372 | -0.00879159 | -27.8390% | 9 | 141 | 0 |
| 04256520 | 150 | 0.050262 | 0.059387 | -0.00912527 | -18.1555% | 9 | 141 | 0 |
| 04379243 | 150 | 0.036284 | 0.045803 | -0.00951908 | -26.2349% | 3 | 147 | 0 |
| 04530566 | 150 | 0.033204 | 0.040565 | -0.00736094 | -22.1689% | 4 | 146 | 0 |
