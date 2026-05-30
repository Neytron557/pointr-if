# Third Pass SEED-PoinTr-IF Report

Date: 2026-05-30

## Data/checkpoint status

- Real dataset: `data/real_projected_shapenet55_adapointr_predictions/manifests/{train,val,test}_triplets.csv`.
- Split sizes: train 1200, val 150, test 150.
- Local PoinTr checkpoints found: `external/PoinTr/pretrained/AdaPoinTr_PCN.pth` and `external/PoinTr/pretrained/AdaPoinTr_ps55.pth`.
- No original PoinTr `.pth` checkpoint was present locally. Original PoinTr configs exist under `external/PoinTr/cfgs/*/PoinTr.yaml`, but no fair original-PoinTr held-out run was produced.
- All final claims below use real held-out validation/test samples only. Synthetic smoke or ablation outputs are not used as final evidence.

## Protocol audit

Point-count audit showed the old second-pass protocol was discarding output points:

| split | rows | partial min/median/max | Ada coarse min/median/max | GT min/median/max |
|---|---:|---:|---:|---:|
| train | 1200 | 47 / 2356.5 / 10127 | 8192 / 8192 / 8192 | 8192 / 8192 / 8192 |
| val | 150 | 236 / 2270.5 / 10236 | 8192 / 8192 / 8192 | 8192 / 8192 / 8192 |
| test | 150 | 245 / 2763.0 / 11830 | 8192 / 8192 / 8192 | 8192 / 8192 / 8192 |

Chosen primary protocol for this pass:

- `--resample-mode fps`
- `--n-partial 2048`
- `--n-coarse 4096`
- `--n-gt 4096`
- `--n-output 4096`
- `--eval-seed 200570`

Under this protocol, AdaPoinTr coarse on the 150 held-out test samples is:

| method | n | CD down | F-score up |
|---|---:|---:|---:|
| Partial input | 150 | 0.140238 | 0.272746 |
| AdaPoinTr coarse | 150 | 0.044413 | 0.593049 |
| Anchor partial+coarse FPS | 150 | 0.047418 | 0.529111 |

The second-pass headline result remains real under its old protocol, but should not be mixed with this pass:

| method | protocol | n | CD down | F-score up | CD gain |
|---|---|---:|---:|---:|---:|
| AdaPoinTr coarse | old random/downsampled eval | 150 | 0.069386 | 0.274315 | 0.0000% |
| GMV-PoinTr-IF seed 570 | old random/downsampled eval | 150 | 0.069101 | 0.277758 | 0.4107% |

Re-running GMV seed 570 under the chosen FPS-4096 protocol produced only a smaller, non-significant gain:

| method | n | CD down | F-score up | CD gain vs Ada |
|---|---:|---:|---:|---:|
| GMV-PoinTr-IF seed 570 | 150 | 0.044374 | 0.592738 | 0.0898% |

Paired GMV stats under the chosen protocol: bootstrap 95% CI for CD gain `[-0.2696%, 0.4789%]`, paired t-test `p=0.6356`, Wilcoxon `p=0.6052`, positive/negative/zero samples `69/81/0`.

## Candidate-bank oracle

Implemented candidate-bank support and oracle evaluation with:

- identity AdaPoinTr coarse candidates,
- deterministic partial/coarse mirror symmetry candidates over x/y/z,
- existing GMV/gated/naive learned outputs when requested,
- concatenation+FPS candidate fusion baseline,
- sample-level oracle and optional point-level oracle approximation.

The final seedbank used only AdaPoinTr identity plus deterministic symmetry candidates at 4096 candidate points. On the held-out test set:

| candidate-bank method | n | CD down | F-score up | CD gain vs Ada |
|---|---:|---:|---:|---:|
| AdaPoinTr coarse | 150 | 0.044413 | 0.593049 | 0.0000% |
| AdaPoinTr identity | 150 | 0.045454 | 0.563569 | -2.3424% |
| all candidates FPS | 150 | 0.080444 | 0.365003 | -81.1260% |
| sample oracle, non-deployable | 150 | 0.043258 | 0.592184 | 2.6005% |

The sample oracle selected AdaPoinTr coarse for 109/150 samples, AdaPoinTr identity for 19/150, symmetry candidates for 20/150, and all-candidates FPS for 2/150. This clears the >1% headroom gate, but the best non-oracle individual sources are still worse than AdaPoinTr coarse.

Validation seedbank oracle also had headroom: AdaPoinTr coarse CD `0.046946`, sample-oracle CD `0.044740`, which is a `4.6993%` oracle gain.

## Experiment matrix results

Completed real experiments:

| ID | method | split | n | protocol | CD down | F-score up | CD gain vs Ada |
|---|---|---|---:|---|---:|---:|---:|
| A1 | AdaPoinTr coarse | test | 150 | FPS 2048/4096/4096/4096 | 0.044413 | 0.593049 | 0.0000% |
| A3 | GMV-PoinTr-IF seed 570 | test | 150 | FPS 2048/4096/4096/4096 | 0.044374 | 0.592738 | 0.0898% |
| B6 | symmetry + Ada seedbank source policy | test | 150 | FPS 2048/4096/4096/4096 | 0.044413 | 0.593049 | 0.0000% |
| B-oracle | seedbank sample oracle, non-deployable | test | 150 | FPS 4096 candidates/GT/output | 0.043258 | 0.592184 | 2.6005% |
| C2/C3 safe | SEED-PoinTr-IF passthrough seed 570 | test | 150 | FPS 2048/4096/4096/4096 | 0.044413 | 0.593049 | 0.0000% |

Learned SEED attempts:

| run | val n | result |
|---|---:|---|
| `seed570_bounded` | 50 | epoch 1 val CD worsened from 0.068399 to 0.079076, `-15.61%`; stopped. |
| `seed570_passthrough` | 50 | epochs 1-2 preserved AdaPoinTr exactly, `0.00%`; used for held-out test because it was stable. |
| `seed570_candidate_entry` | 50 | epoch 1 val CD worsened from 0.058564 to 0.080349, `-37.20%`; stopped. |

The validation-learned category source policy chose AdaPoinTr coarse for every held-out category that passed the minimum validation support threshold. Its held-out result is exactly the AdaPoinTr baseline.

## Best final method

Best deployable method under the chosen protocol remains GMV-PoinTr-IF seed 570, but its gain is only `0.0898%` and is not statistically supported. SEED-PoinTr-IF did not beat AdaPoinTr on held-out test; the stable SEED checkpoint is a baseline-preserving passthrough.

## Whether this qualifies as strong success

No. The candidate bank has real non-deployable headroom (`2.6005%` test CD oracle gain), so the third-pass direction is plausible, but neither the learned SEED refiner nor a validation-learned source policy recovered that headroom. The project should still be described as a small, honest refinement effort rather than a strong benchmark improvement over AdaPoinTr.

## Artifact paths

Code and configs:

- `src/pointr_if/point_ops.py`
- `src/pointr_if/datasets.py`
- `src/pointr_if/evaluate.py`
- `src/pointr_if/models.py`
- `src/pointr_if/train_refiner.py`
- `configs/real_projected_shapenet55_seed_if.yaml`
- `tools/audit_point_counts.py`
- `tools/build_candidate_manifest.py`
- `tools/evaluate_candidate_bank_oracle.py`
- `tools/evaluate_candidate_source_policy.py`
- `tools/export_pointr_tta_predictions.py`
- `scripts/audit_current_results.sh`
- `scripts/evaluate_candidate_oracle.sh`
- `scripts/evaluate_candidate_source_policy.sh`
- `scripts/evaluate_seed_if.sh`
- `scripts/export_tta_candidates.sh`
- `scripts/run_third_pass_sweep.sh`
- `scripts/train_seed_if.sh`
- `tests/test_protocol_and_seed_if.py`

Reports and metrics:

- `reports/third_pass_protocol_audit/`
- `outputs/seed_pointr_if/test_candidate_oracle_seedbank/oracle_summary.json`
- `outputs/seed_pointr_if/test_candidate_oracle_seedbank/per_sample_metrics.csv`
- `outputs/seed_pointr_if/test_candidate_oracle_seedbank/stats/paired_stats.md`
- `outputs/seed_pointr_if/val_candidate_oracle_seedbank/oracle_summary.json`
- `outputs/seed_pointr_if/category_source_policy_seedbank/source_policy_summary.md`
- `outputs/seed_pointr_if/seed570_passthrough/best_model.pt`
- `outputs/seed_pointr_if/seed570_passthrough/metrics.csv`
- `outputs/seed_pointr_if/seed570_passthrough/test_eval/metrics.json`
- `outputs/seed_pointr_if/seed570_passthrough/test_eval/stats/paired_stats.md`
- `outputs/real_projected_shapenet55_gmv_if/test_eval_fps4096/metrics.json`
- `outputs/real_projected_shapenet55_gmv_if/test_eval_fps4096/stats/paired_stats.md`

## Honest limitations

- No original PoinTr checkpoint was available locally, so original-PoinTr comparison rows are absent.
- AdaPoinTr is a strong baseline in this real export; most candidate sources are worse in aggregate.
- The sample oracle is non-deployable because it uses ground-truth test CD to choose a source.
- The learned SEED model did implement candidate fusion, dense output, expansion, candidate confidence, and hard-example weighting, but the tested configurations either collapsed or preserved AdaPoinTr exactly.
- The final SEED run used a deliberately conservative passthrough score to avoid reporting a degraded model as an improvement.
- TTA export support was implemented, but full AdaPoinTr TTA inference was not run in the final real table because the symmetry/identity seedbank already cleared the oracle gate and the learned refiner failed to exploit it.
