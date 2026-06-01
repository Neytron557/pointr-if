import torch
from torch.utils.data import DataLoader

from pointr_if.datasets import SyntheticCompletionDataset
from pointr_if.models import GatedMultiViewSelfStructureRefiner, ImplicitSurfaceRefiner, build_model_from_config
from pointr_if.point_ops import chamfer_distance, sample_occupancy_queries
from pointr_if.train_refiner import compute_losses


def test_forward_shapes():
    dset = SyntheticCompletionDataset(num_samples=2, n_partial=16, n_coarse=32, n_gt=32, seed=0)
    batch = next(iter(DataLoader(dset, batch_size=2)))
    model = ImplicitSurfaceRefiner(hidden_dim=16, point_feature_dim=12, global_dim=24, fourier_bands=2, k=4)
    query, labels = sample_occupancy_queries(batch["gt"].float(), n_query=10)
    out = model(batch["partial"].float(), batch["coarse"].float(), query.float())
    assert out["refined"].shape == batch["coarse"].shape
    assert out["occ_logits"].shape == labels.shape
    assert torch.isfinite(chamfer_distance(out["refined"], batch["gt"].float()))


def test_gmv_refiner_shapes_and_identity_initialization():
    dset = SyntheticCompletionDataset(num_samples=2, n_partial=16, n_coarse=24, n_gt=24, seed=1)
    batch = next(iter(DataLoader(dset, batch_size=2)))
    model = GatedMultiViewSelfStructureRefiner(
        hidden_dim=16,
        point_feature_dim=12,
        global_dim=20,
        fourier_bands=1,
        k=4,
        delta_scale=0.04,
        multiview_resolution=16,
        multiview_feature_dim=8,
    )
    query, labels = sample_occupancy_queries(batch["gt"].float(), n_query=7)
    out = model(batch["partial"].float(), batch["coarse"].float(), query.float())

    assert out["refined"].shape == batch["coarse"].shape
    assert out["gate"].shape == batch["coarse"].shape[:2]
    assert out["occ_logits"].shape == labels.shape
    assert torch.all(out["gate"] >= 0)
    assert torch.all(out["gate"] <= 1)
    assert torch.allclose(out["refined"], batch["coarse"].float(), atol=1e-6)


def test_model_builder_dispatches_gmv_type():
    model = build_model_from_config(
        {
            "model": {
                "type": "gmv",
                "hidden_dim": 8,
                "point_feature_dim": 6,
                "global_dim": 10,
                "fourier_bands": 1,
                "k": 2,
                "multiview_resolution": 8,
                "multiview_feature_dim": 4,
            }
        }
    )

    assert isinstance(model, GatedMultiViewSelfStructureRefiner)


def test_feature_conditioned_refiner_forward_tiny_backbone():
    model = build_model_from_config(
        {
            "model": {
                "type": "feature_gmv",
                "hidden_dim": 32,
                "point_feature_dim": 16,
                "global_dim": 24,
                "fourier_bands": 2,
                "k": 2,
                "use_multiview": True,
                "multiview_resolution": 16,
                "multiview_feature_dim": 8,
                "multiview_backbone": "tiny",
                "coarse_feature_dim": 12,
                "coarse_feature_embed_dim": 8,
            }
        }
    )
    partial = torch.randn(1, 8, 3)
    coarse = torch.randn(1, 12, 3)
    features = torch.randn(1, 12, 12)

    out = model(partial, coarse, coarse_features=features)

    assert out["refined"].shape == coarse.shape


def test_gmv_compute_losses_includes_gate_terms():
    dset = SyntheticCompletionDataset(num_samples=2, n_partial=12, n_coarse=16, n_gt=16, seed=2)
    batch = next(iter(DataLoader(dset, batch_size=2)))
    batch = {k: v.float() if torch.is_tensor(v) else v for k, v in batch.items()}
    model = build_model_from_config(
        {
            "model": {
                "type": "gmv",
                "hidden_dim": 12,
                "point_feature_dim": 8,
                "global_dim": 10,
                "fourier_bands": 1,
                "k": 3,
                "multiview_resolution": 8,
                "multiview_feature_dim": 4,
            }
        }
    )
    cfg = {
        "queries": {"n_query": 5},
        "loss": {
            "lambda_cd": 1.0,
            "lambda_occ": 0.0,
            "lambda_delta": 0.1,
            "lambda_partial": 0.0,
            "lambda_gate_bce": 0.05,
            "lambda_gate_sparsity": 0.01,
            "lambda_delta_l2": 0.02,
        },
    }

    loss, logs = compute_losses(model, batch, cfg)

    assert torch.isfinite(loss)
    assert "loss_gate_bce" in logs
    assert "loss_gate_sparsity" in logs
    assert "loss_delta_l2" in logs
