from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def mlp(channels, activation=nn.ReLU, final_activation: Optional[nn.Module] = None) -> nn.Sequential:
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        if i != len(channels) - 2:
            layers.append(activation())
        elif final_activation is not None:
            layers.append(final_activation)
    return nn.Sequential(*layers)


class FourierFeatures(nn.Module):
    """Sin/cos positional encoding for 3D query points."""

    def __init__(self, num_bands: int = 6, include_xyz: bool = True):
        super().__init__()
        self.num_bands = int(num_bands)
        self.include_xyz = include_xyz
        if self.num_bands > 0:
            bands = 2.0 ** torch.arange(self.num_bands).float()
            self.register_buffer("bands", bands)
        else:
            self.register_buffer("bands", torch.empty(0))

    @property
    def out_dim(self) -> int:
        return (3 if self.include_xyz else 0) + 3 * 2 * self.num_bands

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        outs = []
        if self.include_xyz:
            outs.append(xyz)
        if self.num_bands > 0:
            scaled = xyz.unsqueeze(-1) * self.bands  # [B,N,3,K]
            scaled = scaled.flatten(-2)  # [B,N,3K]
            outs.extend([torch.sin(torch.pi * scaled), torch.cos(torch.pi * scaled)])
        return torch.cat(outs, dim=-1)


@dataclass
class EncodedPointSet:
    base_points: torch.Tensor
    point_features: torch.Tensor
    global_feature: torch.Tensor


class PointSetEncoder(nn.Module):
    """Lightweight PointNet-style encoder for partial + coarse points.

    Each point gets a source flag: 0 for observed partial point, 1 for coarse
    completion point. The global max pooled code carries shape context; the per
    point features support local implicit queries.
    """

    def __init__(self, hidden_dim: int = 96, point_feature_dim: int = 96, global_dim: int = 192):
        super().__init__()
        self.point_mlp = mlp([4, hidden_dim, point_feature_dim, point_feature_dim])
        self.global_mlp = mlp([point_feature_dim, global_dim, global_dim])

    def forward(self, partial: torch.Tensor, coarse: torch.Tensor) -> EncodedPointSet:
        b = partial.shape[0]
        p_flag = torch.zeros(b, partial.shape[1], 1, device=partial.device, dtype=partial.dtype)
        c_flag = torch.ones(b, coarse.shape[1], 1, device=coarse.device, dtype=coarse.dtype)
        base = torch.cat([partial, coarse], dim=1)
        src = torch.cat([p_flag, c_flag], dim=1)
        point_in = torch.cat([base, src], dim=-1)
        feats = self.point_mlp(point_in)
        global_feat = feats.max(dim=1).values
        global_feat = self.global_mlp(global_feat)
        return EncodedPointSet(base_points=base, point_features=feats, global_feature=global_feat)


class ImplicitSurfaceRefiner(nn.Module):
    """Implicit occupancy + residual point refiner for PoinTr predictions.

    The model is trained with two compatible objectives:
      1. Predict occupancy / near-surface labels at arbitrary 3D query locations.
      2. Predict residual displacements for coarse PoinTr points, yielding a refined
         complete point cloud that can be evaluated directly with Chamfer distance.

    This keeps the proposal's implicit-function idea while staying feasible without
    watertight mesh occupancy labels.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        point_feature_dim: int = 96,
        global_dim: int = 192,
        fourier_bands: int = 6,
        k: int = 12,
        use_local: bool = True,
        delta_scale: float = 0.12,
    ):
        super().__init__()
        self.k = int(k)
        self.use_local = bool(use_local)
        self.delta_scale = float(delta_scale)
        self.encoder = PointSetEncoder(hidden_dim=hidden_dim, point_feature_dim=point_feature_dim, global_dim=global_dim)
        self.fourier = FourierFeatures(num_bands=fourier_bands, include_xyz=True)

        if self.use_local:
            self.local_mlp = mlp([point_feature_dim + 4, hidden_dim, hidden_dim])
            local_dim = hidden_dim
        else:
            local_dim = 0

        query_in_dim = self.fourier.out_dim + global_dim + local_dim
        self.query_trunk = mlp([query_in_dim, hidden_dim, hidden_dim, hidden_dim])
        self.occ_head = nn.Linear(hidden_dim, 1)
        self.delta_head = nn.Linear(hidden_dim, 3)

    def _local_features(self, query: torch.Tensor, encoded: EncodedPointSet) -> torch.Tensor:
        base = encoded.base_points
        feats = encoded.point_features
        k = min(max(self.k, 1), base.shape[1])
        d = torch.cdist(query, base, p=2)
        knn_d, idx = d.topk(k=k, dim=2, largest=False)

        idx_feat = idx.unsqueeze(-1).expand(-1, -1, -1, feats.shape[-1])
        feats_exp = feats.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
        neigh_feats = feats_exp.gather(dim=2, index=idx_feat)

        idx_xyz = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        base_exp = base.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
        neigh_xyz = base_exp.gather(dim=2, index=idx_xyz)
        rel = query.unsqueeze(2) - neigh_xyz
        inp = torch.cat([neigh_feats, rel, knn_d.unsqueeze(-1)], dim=-1)
        local_tokens = self.local_mlp(inp)
        weights = torch.softmax(-knn_d.clamp_min(1e-6), dim=2).unsqueeze(-1)
        return (local_tokens * weights).sum(dim=2)

    def query_field(self, query: torch.Tensor, partial: torch.Tensor, coarse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(partial, coarse)
        return self.query_field_encoded(query, encoded)

    def query_field_encoded(self, query: torch.Tensor, encoded: EncodedPointSet) -> Tuple[torch.Tensor, torch.Tensor]:
        b, nq, _ = query.shape
        global_feat = encoded.global_feature.unsqueeze(1).expand(-1, nq, -1)
        query_feat = [self.fourier(query), global_feat]
        if self.use_local:
            query_feat.append(self._local_features(query, encoded))
        h = self.query_trunk(torch.cat(query_feat, dim=-1))
        occ_logits = self.occ_head(h).squeeze(-1)
        delta = torch.tanh(self.delta_head(h)) * self.delta_scale
        return occ_logits, delta

    def forward(
        self,
        partial: torch.Tensor,
        coarse: torch.Tensor,
        query: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(partial, coarse)
        coarse_occ, coarse_delta = self.query_field_encoded(coarse, encoded)
        refined = coarse + coarse_delta
        out = {"refined": refined, "coarse_delta": coarse_delta, "coarse_occ_logits": coarse_occ}
        if query is not None:
            occ_logits, query_delta = self.query_field_encoded(query, encoded)
            out.update({"occ_logits": occ_logits, "query_delta": query_delta})
        return out


def _project_canonical(points: torch.Tensor, view: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project points to one of six canonical orthographic views.

    Returns normalized grid coordinates in [-1, 1] and a signed depth channel.
    """
    x, y, z = points.unbind(dim=-1)
    if view == 0:  # +x
        return torch.stack([y, z], dim=-1), x
    if view == 1:  # -x
        return torch.stack([-y, z], dim=-1), -x
    if view == 2:  # +y
        return torch.stack([x, z], dim=-1), y
    if view == 3:  # -y
        return torch.stack([-x, z], dim=-1), -y
    if view == 4:  # +z
        return torch.stack([x, y], dim=-1), z
    if view == 5:  # -z
        return torch.stack([x, -y], dim=-1), -z
    raise ValueError(f"Unsupported canonical view index: {view}")


def _rasterize_multiview(partial: torch.Tensor, coarse: torch.Tensor, resolution: int) -> torch.Tensor:
    """Rasterize partial/coarse clouds into simple six-view occupancy/depth maps."""
    b = partial.shape[0]
    res = int(resolution)
    maps = partial.new_zeros((b, 6, 4, res, res))
    clouds = ((partial, 0), (coarse, 2))
    with torch.no_grad():
        for view in range(6):
            for points, channel_offset in clouds:
                uv, depth = _project_canonical(points[..., :3].clamp(-1.0, 1.0), view)
                pix = ((uv.clamp(-1.0, 1.0) + 1.0) * 0.5 * (res - 1)).round().long()
                x_idx = pix[..., 0].clamp(0, res - 1)
                y_idx = pix[..., 1].clamp(0, res - 1)
                flat_idx = y_idx * res + x_idx
                ones = torch.ones_like(depth)
                for batch_idx in range(b):
                    count_flat = maps[batch_idx, view, channel_offset].flatten()
                    depth_flat = maps[batch_idx, view, channel_offset + 1].flatten()
                    count_flat.index_add_(0, flat_idx[batch_idx], ones[batch_idx])
                    depth_flat.index_add_(0, flat_idx[batch_idx], depth[batch_idx])
        for channel_offset in (0, 2):
            counts = maps[:, :, channel_offset : channel_offset + 1]
            depth_sum = maps[:, :, channel_offset + 1 : channel_offset + 2]
            maps[:, :, channel_offset + 1 : channel_offset + 2] = torch.where(counts > 0, depth_sum / counts.clamp_min(1.0), depth_sum)
            maps[:, :, channel_offset : channel_offset + 1] = torch.log1p(counts)
    return maps


class GatedMultiViewSelfStructureRefiner(nn.Module):
    """Conservative gated refiner with local geometry and self-view map features."""

    def __init__(
        self,
        hidden_dim: int = 128,
        point_feature_dim: int = 96,
        global_dim: int = 192,
        fourier_bands: int = 4,
        k: int = 12,
        use_local: bool = True,
        delta_scale: float = 0.05,
        gate_bias: float = -2.0,
        use_multiview: bool = True,
        multiview_resolution: int = 32,
        multiview_feature_dim: int = 32,
    ):
        super().__init__()
        self.k = int(k)
        self.use_local = bool(use_local)
        self.delta_scale = float(delta_scale)
        self.use_multiview = bool(use_multiview)
        self.multiview_resolution = int(multiview_resolution)
        self.multiview_feature_dim = int(multiview_feature_dim) if self.use_multiview else 0
        self.encoder = PointSetEncoder(hidden_dim=hidden_dim, point_feature_dim=point_feature_dim, global_dim=global_dim)
        self.fourier = FourierFeatures(num_bands=fourier_bands, include_xyz=True)

        if self.use_local:
            self.local_mlp = mlp([point_feature_dim + 4, hidden_dim, hidden_dim])
            local_dim = hidden_dim
        else:
            local_dim = 0

        if self.use_multiview:
            self.view_cnn = nn.Sequential(
                nn.Conv2d(4, hidden_dim // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim // 2, self.multiview_feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.view_cnn = None

        geom_dim = 3
        trunk_dim = self.fourier.out_dim + global_dim + local_dim + self.multiview_feature_dim + geom_dim
        self.query_trunk = mlp([trunk_dim, hidden_dim, hidden_dim, hidden_dim])
        self.occ_head = nn.Linear(hidden_dim, 1)
        self.delta_head = nn.Linear(hidden_dim, 3)
        self.gate_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias))

    def _local_features(self, query: torch.Tensor, encoded: EncodedPointSet) -> torch.Tensor:
        base = encoded.base_points
        feats = encoded.point_features
        k = min(max(self.k, 1), base.shape[1])
        d = torch.cdist(query, base, p=2)
        knn_d, idx = d.topk(k=k, dim=2, largest=False)
        idx_feat = idx.unsqueeze(-1).expand(-1, -1, -1, feats.shape[-1])
        feats_exp = feats.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
        neigh_feats = feats_exp.gather(dim=2, index=idx_feat)
        idx_xyz = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        base_exp = base.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
        neigh_xyz = base_exp.gather(dim=2, index=idx_xyz)
        rel = query.unsqueeze(2) - neigh_xyz
        local_tokens = self.local_mlp(torch.cat([neigh_feats, rel, knn_d.unsqueeze(-1)], dim=-1))
        weights = torch.softmax(-knn_d.clamp_min(1e-6), dim=2).unsqueeze(-1)
        return (local_tokens * weights).sum(dim=2)

    def _geometry_features(self, query: torch.Tensor, partial: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        nearest_partial = torch.cdist(query, partial, p=2).min(dim=2).values
        nearest_coarse = torch.cdist(query, coarse, p=2).min(dim=2).values
        radius = torch.linalg.norm(query, dim=-1)
        return torch.stack([nearest_partial, nearest_coarse, radius], dim=-1)

    def _encode_views(self, partial: torch.Tensor, coarse: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_multiview or self.view_cnn is None:
            return None
        maps = _rasterize_multiview(partial, coarse, self.multiview_resolution)
        b = maps.shape[0]
        maps = maps.reshape(b * 6, 4, self.multiview_resolution, self.multiview_resolution)
        feats = self.view_cnn(maps)
        return feats.reshape(b, 6, self.multiview_feature_dim, self.multiview_resolution, self.multiview_resolution)

    def _sample_view_features(self, query: torch.Tensor, view_feats: Optional[torch.Tensor]) -> torch.Tensor:
        if view_feats is None:
            return query.new_zeros(query.shape[0], query.shape[1], 0)
        b, _, feat_dim, h, w = view_feats.shape
        grids = []
        for view in range(6):
            uv, _ = _project_canonical(query[..., :3].clamp(-1.0, 1.0), view)
            grids.append(uv.clamp(-1.0, 1.0))
        grid = torch.stack(grids, dim=1).reshape(b * 6, query.shape[1], 1, 2)
        sampled = F.grid_sample(view_feats.reshape(b * 6, feat_dim, h, w), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled.squeeze(-1).reshape(b, 6, feat_dim, query.shape[1]).permute(0, 3, 1, 2)
        return sampled.mean(dim=2)

    def _trunk_features(
        self,
        query: torch.Tensor,
        encoded: EncodedPointSet,
        partial: torch.Tensor,
        coarse: torch.Tensor,
        view_feats: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, nq, _ = query.shape
        parts = [
            self.fourier(query),
            encoded.global_feature.unsqueeze(1).expand(-1, nq, -1),
            self._geometry_features(query, partial, coarse),
        ]
        if self.use_local:
            parts.append(self._local_features(query, encoded))
        if self.use_multiview:
            parts.append(self._sample_view_features(query, view_feats))
        return self.query_trunk(torch.cat(parts, dim=-1))

    def forward(
        self,
        partial: torch.Tensor,
        coarse: torch.Tensor,
        query: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(partial, coarse)
        view_feats = self._encode_views(partial, coarse)
        h = self._trunk_features(coarse, encoded, partial, coarse, view_feats)
        raw_delta = torch.tanh(self.delta_head(h)) * self.delta_scale
        gate_logits = self.gate_head(h).squeeze(-1)
        gate = torch.sigmoid(gate_logits)
        coarse_delta = gate.unsqueeze(-1) * raw_delta
        refined = coarse + coarse_delta
        out = {
            "refined": refined,
            "coarse_delta": coarse_delta,
            "raw_delta": raw_delta,
            "gate_logits": gate_logits,
            "gate": gate,
            "coarse_occ_logits": self.occ_head(h).squeeze(-1),
        }
        if query is not None:
            query_h = self._trunk_features(query, encoded, partial, coarse, view_feats)
            out.update(
                {
                    "occ_logits": self.occ_head(query_h).squeeze(-1),
                    "query_delta": torch.tanh(self.delta_head(query_h)) * self.delta_scale,
                }
            )
        return out


class SeedPointrIFRefiner(nn.Module):
    """Dense candidate fusion refiner for SEED-PoinTr-IF.

    The model consumes a bank of candidate completions from AdaPoinTr/TTA/symmetry
    and learned refiners, predicts per-point confidence and residuals, expands each
    candidate point into children, then returns the top-confidence dense cloud.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        global_dim: int = 192,
        source_embed_dim: int = 16,
        fourier_bands: int = 4,
        k: int = 8,
        max_sources: int = 32,
        candidate_points: int = 1024,
        output_points: int = 4096,
        expansion_factor: int = 4,
        delta_scale: float = 0.04,
        child_radius: float = 0.04,
        coarse_passthrough_score: float = 0.0,
    ):
        super().__init__()
        self.k = int(k)
        self.max_sources = int(max_sources)
        self.candidate_points = int(candidate_points)
        self.output_points = int(output_points)
        self.expansion_factor = int(expansion_factor)
        self.delta_scale = float(delta_scale)
        self.child_radius = float(child_radius)
        self.coarse_passthrough_score = float(coarse_passthrough_score)
        self.encoder = PointSetEncoder(hidden_dim=hidden_dim, point_feature_dim=hidden_dim, global_dim=global_dim)
        self.fourier = FourierFeatures(num_bands=fourier_bands, include_xyz=True)
        self.source_embedding = nn.Embedding(self.max_sources, int(source_embed_dim))

        geom_dim = 4
        in_dim = self.fourier.out_dim + global_dim + int(source_embed_dim) + geom_dim
        self.trunk = mlp([in_dim, hidden_dim, hidden_dim, hidden_dim])
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.residual_head = nn.Linear(hidden_dim, 3)
        self.source_head = nn.Linear(hidden_dim, self.max_sources)
        self.child_offset_head = nn.Linear(hidden_dim, self.expansion_factor * 3)
        self.child_confidence_head = nn.Linear(hidden_dim, self.expansion_factor)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.child_offset_head.weight)
        nn.init.zeros_(self.child_offset_head.bias)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.constant_(self.confidence_head.bias, -1.0)
        nn.init.zeros_(self.child_confidence_head.weight)
        nn.init.constant_(self.child_confidence_head.bias, -1.0)

    def _geometry_features(self, points: torch.Tensor, partial: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        nearest_partial = torch.cdist(points, partial, p=2).min(dim=2).values
        nearest_coarse = torch.cdist(points, coarse, p=2).min(dim=2).values
        radius = torch.linalg.norm(points, dim=-1)
        missing_score = (nearest_partial - nearest_coarse).clamp_min(0.0)
        return torch.stack([nearest_partial, nearest_coarse, radius, missing_score], dim=-1)

    def _source_ids(self, candidates: torch.Tensor, source_ids: Optional[torch.Tensor]) -> torch.Tensor:
        b, sources, _points, _xyz = candidates.shape
        if source_ids is None:
            source_ids = torch.arange(sources, device=candidates.device).unsqueeze(0).expand(b, -1)
        source_ids = source_ids.to(device=candidates.device, dtype=torch.long).clamp(0, self.max_sources - 1)
        return source_ids.unsqueeze(-1).expand(-1, -1, candidates.shape[2]).reshape(b, sources * candidates.shape[2])

    def forward(
        self,
        partial: torch.Tensor,
        coarse: torch.Tensor,
        candidates: torch.Tensor,
        source_ids: Optional[torch.Tensor] = None,
        query: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if candidates.ndim != 4 or candidates.shape[-1] != 3:
            raise ValueError("candidates must be [B,S,N,3]")
        b, sources, points_per_source, _ = candidates.shape
        flat = candidates.reshape(b, sources * points_per_source, 3)
        flat_source_ids = self._source_ids(candidates, source_ids)
        encoded = self.encoder(partial, coarse)
        global_feat = encoded.global_feature.unsqueeze(1).expand(-1, flat.shape[1], -1)
        source_feat = self.source_embedding(flat_source_ids)
        geom = self._geometry_features(flat, partial, coarse)
        h = self.trunk(torch.cat([self.fourier(flat), global_feat, source_feat, geom], dim=-1))

        confidence_logits = self.confidence_head(h).squeeze(-1)
        residual_delta = torch.tanh(self.residual_head(h)) * self.delta_scale
        fused = flat + residual_delta
        source_logits = self.source_head(h)

        child_offsets = torch.tanh(self.child_offset_head(h)).reshape(b, flat.shape[1], self.expansion_factor, 3)
        child_offsets = child_offsets * self.child_radius
        child_confidence = self.child_confidence_head(h)
        children = fused.unsqueeze(2) + child_offsets
        child_scores = confidence_logits.unsqueeze(-1) + child_confidence
        dense = children.reshape(b, flat.shape[1] * self.expansion_factor, 3)
        dense_scores = child_scores.reshape(b, flat.shape[1] * self.expansion_factor)
        if coarse.shape[1] > 0:
            dense = torch.cat([coarse, dense], dim=1)
            coarse_scores = dense_scores.new_full((b, coarse.shape[1]), self.coarse_passthrough_score)
            dense_scores = torch.cat([coarse_scores, dense_scores], dim=1)

        if dense.shape[1] < self.output_points:
            repeats = (self.output_points + dense.shape[1] - 1) // dense.shape[1]
            dense = dense.repeat(1, repeats, 1)
            dense_scores = dense_scores.repeat(1, repeats)
        topk = torch.topk(dense_scores, k=self.output_points, dim=1).indices
        refined = dense.gather(1, topk.unsqueeze(-1).expand(-1, -1, 3))
        return {
            "refined": refined,
            "point_confidence": confidence_logits,
            "source_logits": source_logits,
            "residual_delta": residual_delta,
            "child_confidence": child_confidence,
            "candidate_points": fused,
            "dense_candidates": dense,
            "selected_indices": topk,
        }


def build_model_from_config(cfg: Dict) -> nn.Module:
    model_cfg = cfg.get("model", cfg)
    model_type = str(model_cfg.get("type", "implicit")).lower()
    if model_type in {"gmv", "gated_multiview", "gmv_pointr_if"}:
        return GatedMultiViewSelfStructureRefiner(
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            point_feature_dim=int(model_cfg.get("point_feature_dim", 96)),
            global_dim=int(model_cfg.get("global_dim", 192)),
            fourier_bands=int(model_cfg.get("fourier_bands", 4)),
            k=int(model_cfg.get("k", 12)),
            use_local=bool(model_cfg.get("use_local", True)),
            delta_scale=float(model_cfg.get("delta_scale", 0.05)),
            gate_bias=float(model_cfg.get("gate_bias", -2.0)),
            use_multiview=bool(model_cfg.get("use_multiview", True)),
            multiview_resolution=int(model_cfg.get("multiview_resolution", 32)),
            multiview_feature_dim=int(model_cfg.get("multiview_feature_dim", 32)),
        )
    if model_type in {"seed", "seed_if", "seed_pointr_if"}:
        return SeedPointrIFRefiner(
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            global_dim=int(model_cfg.get("global_dim", 192)),
            source_embed_dim=int(model_cfg.get("source_embed_dim", 16)),
            fourier_bands=int(model_cfg.get("fourier_bands", 4)),
            k=int(model_cfg.get("k", 8)),
            max_sources=int(model_cfg.get("max_sources", 32)),
            candidate_points=int(model_cfg.get("candidate_points", 1024)),
            output_points=int(model_cfg.get("output_points", 4096)),
            expansion_factor=int(model_cfg.get("expansion_factor", 4)),
            delta_scale=float(model_cfg.get("delta_scale", 0.04)),
            child_radius=float(model_cfg.get("child_radius", 0.04)),
            coarse_passthrough_score=float(model_cfg.get("coarse_passthrough_score", 0.0)),
        )
    return ImplicitSurfaceRefiner(
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        point_feature_dim=int(model_cfg.get("point_feature_dim", 96)),
        global_dim=int(model_cfg.get("global_dim", 192)),
        fourier_bands=int(model_cfg.get("fourier_bands", 6)),
        k=int(model_cfg.get("k", 12)),
        use_local=bool(model_cfg.get("use_local", True)),
        delta_scale=float(model_cfg.get("delta_scale", 0.12)),
    )
