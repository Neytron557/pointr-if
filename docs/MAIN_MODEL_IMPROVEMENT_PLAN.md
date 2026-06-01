# Implementation Plan: Deep Improvements for PoinTr-IF (Updated)

This document outlines the detailed plan to improve PoinTr-IF. We will execute both proposed architectures (Joint Backbone Fine-Tuning and Feature-Conditioned Post-Processing Refiner), train on both datasets (Projected ShapeNet55/34 and the gated PCN dataset), and strictly adhere to the 12-hour training limit on the 1-A100 GPU server.

---

## 1. Analysis of Current Performance Bottleneck

The current baseline implementation of PoinTr-IF achieves a marginal Chamfer Distance (CD) improvement of **0.0898%** under the correct FPS-4096 protocol. This abysmally small improvement stems from the following design bottlenecks:

1. **Information Isolation (Black-Box Backbone)**:
   The refiner treats the AdaPoinTr backbone as a black box. It only receives the final coordinate point clouds and has to "re-learn" global shape contexts using a simple PointNet encoder. It has no access to the rich transformer embeddings and attention maps computed by AdaPoinTr.
2. **PointNet Geometric Bottleneck**:
   The `PointSetEncoder` uses a simple PointNet architecture (MLP + global max pooling). PointNet struggles to preserve local spatial hierarchies and fine geometric details, leading to flat, noisy local point displacements.
3. **Coarse View Representation**:
   The Gated Multi-View (GMV) module rasterizes point projections into 6 orthographic views at an extremely low resolution of 32x32 using a tiny 2-layer CNN. This is too coarse to capture fine-grained shape boundaries or depth variations.
4. **Frozen Backbone Constraints**:
   Because the transformer backbone is completely frozen in the refiner training, the model cannot adjust its structural outputs (topology, overall skeleton) to correct large errors. It can only apply local coordinate offsets to the predicted coarse points.
5. **Loss Function Limitations**:
   Using Chamfer Distance (CD) L1 alone allows points to cluster around local dense areas, leaving other areas empty. It does not enforce uniform coverage or surface smoothness.

---

## 2. Proposed Improvements & Target Architectures

To explore both directions within the 12-hour limit, we will prepare and train two separate runs:

### Run A: Unfrozen MVC Fine-Tuning (Joint End-to-End Backbone Optimization)
*   **Concept**: Instead of running a separate post-processing network, we unfreeze the query generator, decoder, and last encoder blocks of the AdaPoinTr backbone itself and train it with Multi-View Consistency (MVC).
*   **A100 Scaling**: We will unfreeze the last 2 encoder blocks, use a batch size of 16 (on A100), 4 view augmentations per object, and train for 12 epochs. This directly improves the base shape completion capabilities.

### Run B: Backbone-Feature-Conditioned Implicit Refiner
*   **Concept**: We keep AdaPoinTr frozen, but extract its intermediate Transformer Decoder features (shape `[B, 4096, 384]`) corresponding to each predicted point. We feed these features, along with local PointNeXt hierarchical features and high-resolution 2D ResNet-18 multi-view projections (128x128), into the implicit refiner.
*   **A100 Scaling**: We will train for 60 epochs with a batch size of 16, using supervised Chamfer Loss + Earth Mover's Distance (EMD) to enforce point distribution uniformity.

---

## 3. PCN Dataset Download Guide

The PCN (Point Cloud Completion Network) dataset download is protected behind a **Google reCAPTCHA** verify gate, which prevents automated command-line downloads. Below is the step-by-step guide to download and configure it manually:

### Step 1: Request Download Link
1.  On your local machine with a web browser, go to:
    `https://gateway.infinitescript.com/?fileName=ShapeNetCompletion`
2.  Solve the Google reCAPTCHA challenge.
3.  Enter your email to receive download links, or click the direct download link displayed after solving the captcha.
4.  Alternatively, use the Baidu Netdisk mirror:
    *   **URL**: `https://pan.baidu.com/s/1Oj-2F_eHMopLF2CWnd8T3A`
    *   **Extraction Code**: `hg24`

### Step 2: Transfer and Extract to Server
1.  Download the `ShapeNetCompletion.zip` archive.
2.  Transfer the zip archive to your server under:
    `/home/ubuntu/ai_and_ml/pointr_if_project/external/PoinTr/data/`
3.  Unzip the file:
    ```bash
    cd /home/ubuntu/ai_and_ml/pointr_if_project/external/PoinTr/data
    unzip ShapeNetCompletion.zip -d PCN
    ```
4.  Ensure the directory structure matches the following (as required by `DATASET.md`):
    ```text
    PoinTr/data/PCN/
    ├── train/
    │   ├── complete/
    │   └── partial/
    ├── val/
    │   ├── complete/
    │   └── partial/
    ├── test/
    │   ├── complete/
    │   └── partial/
    ├── PCN.json
    └── category.txt
    ```

---

## 4. Runbook for 1-A100 Server (Time-Budgeted to 11.5 hours)

This runbook trains both models on the server.

```
[Setup & Verify] -> [Run A: Joint MVC Fine-Tune] -> [Run B: Feature Refiner] -> [Eval & Comparison]
   0.5 hours              4.5 hours                   5.5 hours                  1.0 hour
```

### Step 1: System Check (30 mins)
Verify environment and GPU:
```bash
cd /home/ubuntu/ai_and_ml/pointr_if_project
source .venv/bin/activate
python scripts/doctor.py
```

### Step 2: Train Run A: Joint MVC Backbone Fine-Tuning (4.5 hours)
Run the unfreezing fine-tune script unfreezing the last encoder block and the entire decoder of AdaPoinTr under multi-view consistency.
```bash
UNFREEZE=decoder_plus_last_encoder \
UNFREEZE_LAST_ENCODER_BLOCKS=1 \
LR_DECODER=2e-5 \
LR_ENCODER=1e-6 \
EPOCHS=12 \
VIEWS_PER_OBJECT=4 \
LAMBDA_SELF=0.08 \
bash scripts/train_mvc_pointr_if.sh outputs/fourth_pass/adapointr_mvc_a100
```

### Step 3: Train Run B: Feature-Conditioned Refiner (5.5 hours)
Export prediction features first:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_pointr_predictions.py \
  --pointr-root external/PoinTr \
  --config external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml \
  --checkpoint external/PoinTr/pretrained/AdaPoinTr_ps55.pth \
  --data-root external/PoinTr/data/ShapeNet55-34 \
  --split train \
  --split-name train \
  --selected-json data/real_projected_shapenet55_subset/train_selected_members.json \
  --out-root data/real_projected_shapenet55_adapointr_predictions \
  --batch-size 8 \
  --device cuda
```
Train the refiner using PointNeXt and ResNet multi-view features:
```bash
CUDA_VISIBLE_DEVICES=0 python -m pointr_if.train \
  --config configs/real_projected_shapenet55_gmv_if_a100.yaml \
  --train-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/train_triplets.csv \
  --val-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/val_triplets.csv \
  --out-dir outputs/real_projected_shapenet55_gmv_if_a100 \
  --epochs 60 \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda
```

### Step 4: Evaluate & Compare (1 hour)
Run evaluation on both test sets (Projected ShapeNet and PCN) to compare CD and F-score improvements.
```bash
# Evaluate Run A
bash scripts/evaluate_mvc_pointr_if.sh \
  outputs/fourth_pass/adapointr_mvc_a100/ckpt-best.pth \
  outputs/fourth_pass/adapointr_mvc_a100/test_eval \
  test

# Evaluate Run B
CUDA_VISIBLE_DEVICES=0 python -m pointr_if.evaluate \
  --manifest data/real_projected_shapenet55_adapointr_predictions/manifests/test_triplets.csv \
  --checkpoint outputs/real_projected_shapenet55_gmv_if_a100/best_model.pt \
  --out-dir outputs/real_projected_shapenet55_gmv_if_a100/test_eval \
  --batch-size 16 \
  --device cuda
```

---

## 5. Verification Plan

We will verify the effectiveness of the proposed changes using both automated and manual methods:

### Automated Tests
1.  **Unit Tests**: Run tests inside the virtual environment:
    ```bash
    pytest -q
    ```
2.  **Smoke Tests**: Run synthetic model validation checks:
    ```bash
    bash scripts/run_smoke.sh
    ```
3.  **Chamfer Improvement Gate**: Check if the validation-selected refined outputs achieve the success target on the test set:
    *   **Baseline coarse CD**: `0.044413`
    *   **Success Threshold (5% CD Improvement)**: `CD <= 0.04219`
4.  **Paired Statistical Significance**: Check the paired bootstrap 95% confidence interval and paired t-test/Wilcoxon p-values via `analyze_real_results_stats.py` to confirm that the improvement is statistically significant ($p < 0.01$).

### Manual Verification
1.  **Surface Visualization**: Open the generated qualitative figures (e.g., `outputs/real_projected_shapenet55_gmv_if/test_eval/ranked_qualitative.png`) and manually inspect if the refined shape has fewer artifacts, smoother surfaces, and better topology consistency.
2.  **Point Density Uniformity**: Inspect point cloud output files (`.npy`) using a point cloud visualizer (e.g., Open3D or CloudCompare) to check if the point distribution is uniform and follows the true ground-truth boundary without local point-clustering.
