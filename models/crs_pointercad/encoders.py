"""Typed trainable CRS candidate encoders.

FACE/EDGE consume native UV-Net pointer tensors. Other records use typed
semantic fields; arbitrary object strings and persistent ids are excluded.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import PointerType

_CURVE_TYPES = ("Line3D", "Arc3D", "Circle3D", "Ellipse3D", "EllipticalArc3D", "NurbsCurve3D", "Polyline3D")
_GEOMETRY_TYPES = ("Point3D", "Line3D", "Arc3D", "Circle3D", "Ellipse3D", "EllipticalArc3D", "NurbsCurve3D", "Polyline3D", "Plane3D", "PlaneSurface", "CylinderSurface", "ConeSurface", "SphereSurface", "TorusSurface")


def _numbers(value: Any) -> torch.Tensor:
    """Extract explicitly numeric typed fields only."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32).flatten()
    if isinstance(value, bool):
        return torch.tensor([float(value)], dtype=torch.float32)
    if isinstance(value, (int, float)):
        return torch.tensor([float(value)], dtype=torch.float32)
    if isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            if str(key) in {"id", "name", "external_key", "selection", "owner", "type", "role", "roles", "source", "creator_operation"}:
                continue
            parts.append(_numbers(item))
        return torch.cat(parts) if parts else torch.zeros(1)
    if isinstance(value, (list, tuple)):
        parts = [_numbers(item) for item in value]
        return torch.cat(parts) if parts else torch.zeros(1)
    return torch.zeros(0)


def _fixed(values: torch.Tensor, size: int = 64) -> torch.Tensor:
    values = values.flatten()
    if values.numel() < size:
        values = F.pad(values, (0, size - values.numel()))
    return values[:size]


class CandidateEncoder(nn.Module):
    pointer_type: PointerType

    def __init__(self, output_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.projection = nn.Sequential(nn.Linear(64, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def project(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(_fixed(values).to(self.projection[0].weight.device).unsqueeze(0)).squeeze(0)

    def forward(self, semantic: Any, *, metadata: Mapping[str, Any] | None = None) -> torch.Tensor:
        return self.project(_numbers(semantic))


class NativePointerEncoder(nn.Module):
    """The production FACE/EDGE path; embeddings come from UV-Net."""
    pointer_type: PointerType

    def forward(self, semantic: Any, *, metadata: Mapping[str, Any] | None = None) -> torch.Tensor:
        metadata = metadata or {}
        value = metadata.get("native_embedding")
        if value is None and isinstance(semantic, torch.Tensor) and semantic.numel() == 128:
            value = semantic
        if value is None:
            raise ValueError(f"native {self.pointer_type.value} embedding is required")
        value = value.to(dtype=torch.float32).flatten()
        if value.numel() != 128:
            raise ValueError("native pointer embedding must be 128-D")
        return value


class FaceCandidateEncoder(NativePointerEncoder):
    pointer_type = PointerType.FACE


class EdgeCandidateEncoder(NativePointerEncoder):
    pointer_type = PointerType.EDGE


class Curve3DEncoder(CandidateEncoder):
    """Shared typed curve encoder with variable point/control-point fields."""

    def __init__(self, output_dim: int = 128):
        super().__init__(output_dim=output_dim)
        self.type_embedding = nn.Embedding(len(_CURVE_TYPES), 16)
        self.sequence_projection = nn.Linear(4, 64)
        self.sequence = nn.GRU(64, 64, batch_first=True)
        self.curve_projection = nn.Sequential(nn.Linear(80, 128), nn.LayerNorm(128), nn.GELU())

    @staticmethod
    def _ordered_scalars(value: Any, field_code: float = 0.0):
        """Return an ordered scalar stream without truncating variable lists."""
        if isinstance(value, Mapping):
            for index, (key, item) in enumerate(value.items()):
                if str(key) == "type":
                    continue
                yield from Curve3DEncoder._ordered_scalars(item, field_code=float(index + 1))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from Curve3DEncoder._ordered_scalars(item, field_code=field_code + index * 0.001)
        elif isinstance(value, bool):
            yield float(value), field_code
        elif isinstance(value, (int, float)):
            yield float(value), field_code

    def forward(self, semantic: Mapping[str, Any], *, metadata=None) -> torch.Tensor:
        kind = str(semantic.get("type", ""))
        if kind not in _CURVE_TYPES:
            raise ValueError(f"unsupported Curve3D type: {kind!r}")
        device = self.type_embedding.weight.device
        type_vec = self.type_embedding(torch.tensor([_CURVE_TYPES.index(kind)], device=device)).squeeze(0)
        stream = list(self._ordered_scalars(semantic))
        if not stream:
            stream = [(0.0, 0.0)]
        rows = []
        length = max(1, len(stream) - 1)
        for index, (value, field_code) in enumerate(stream):
            rows.append(torch.tensor([value, field_code, index / length, 1.0], dtype=torch.float32, device=device))
        encoded, _ = self.sequence(self.sequence_projection(torch.stack(rows)).unsqueeze(0))
        sequence_vec = encoded[:, -1, :].squeeze(0)
        return self.curve_projection(torch.cat((type_vec, sequence_vec)).unsqueeze(0)).squeeze(0)


class ProfileCandidateEncoder(nn.Module):
    pointer_type = PointerType.PROFILE

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.curve = Curve3DEncoder(output_dim)
        self.loop_projection = nn.Linear(129, 128)
        self.sequence = nn.GRU(128, 128, batch_first=True)
        self.curve_sequence = nn.GRU(128, 128, batch_first=True)
        self.frame_projection = nn.Linear(64, 128)
        self.output = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, output_dim))

    def forward(self, semantic: Mapping[str, Any], *, metadata=None) -> torch.Tensor:
        loops = semantic.get("loops", ())
        loop_rows = []
        for loop in loops:
            curves = tuple(loop.get("curves", loop.get("profile_curves", ())))
            curve_rows = [self.curve(curve) for curve in curves]
            if not curve_rows:
                raise ValueError("profile loop must contain ordered curves")
            ordered_curves, _ = self.curve_sequence(torch.stack(curve_rows).unsqueeze(0))
            outer = torch.tensor([float(loop.get("is_outer", False))], device=ordered_curves.device)
            loop_rows.append(self.loop_projection(torch.cat((ordered_curves[:, -1, :].squeeze(0), outer)).unsqueeze(0)).squeeze(0))
        if not loop_rows:
            raise ValueError("profile must contain at least one loop")
        ordered, _ = self.sequence(torch.stack(loop_rows).unsqueeze(0))
        frame = self.frame_projection(_fixed(_numbers(semantic.get("transform", semantic.get("frame", {})))).to(self.frame_projection.weight.device).unsqueeze(0)).squeeze(0)
        return self.output(torch.cat((ordered[:, -1, :].squeeze(0), frame)).unsqueeze(0)).squeeze(0)


class SketchReferenceCandidateEncoder(CandidateEncoder):
    pointer_type = PointerType.SKETCH_REFERENCE

    def __init__(self, output_dim: int = 128):
        super().__init__(output_dim)
        self.geometry = nn.Embedding(len(_GEOMETRY_TYPES), 16)
        self.role = nn.Embedding(8, 8)
        self.output = nn.Sequential(nn.Linear(88, 192), nn.GELU(), nn.Linear(192, output_dim))

    def forward(self, semantic: Mapping[str, Any], *, metadata=None) -> torch.Tensor:
        geometry = semantic.get("geometry", semantic)
        kind = str(semantic.get("geometry_type", semantic.get("type", geometry.get("type", ""))))
        if kind not in _GEOMETRY_TYPES:
            raise ValueError(f"unsupported sketch reference geometry type: {kind!r}")
        role_names = ("construction", "feature_input", "projected", "centerline", "profile", "axis", "direction", "other")
        roles = semantic.get("roles", [semantic.get("role", "construction")])
        device = self.geometry.weight.device
        role_rows = [self.role(torch.tensor([role_names.index(str(role)) if str(role) in role_names else 7], device=device)) for role in roles]
        role_vec = torch.stack(role_rows).mean(0).squeeze(0) if role_rows else torch.zeros(8, device=device)
        fields = _fixed(_numbers(geometry), 64).to(device)
        return self.output(torch.cat((self.geometry(torch.tensor([_GEOMETRY_TYPES.index(kind)], device=device)).squeeze(0), role_vec, fields)).unsqueeze(0)).squeeze(0)


class ResolvedGeometryCandidateEncoder(CandidateEncoder):
    pointer_type = PointerType.RESOLVED_GEOMETRY

    def __init__(self, output_dim: int = 128):
        super().__init__(output_dim)
        self.geometry = nn.Embedding(len(_GEOMETRY_TYPES), 24)
        self.output = nn.Sequential(nn.Linear(88, 192), nn.GELU(), nn.Linear(192, output_dim))

    def forward(self, semantic: Mapping[str, Any], *, metadata=None) -> torch.Tensor:
        kind = str(semantic.get("type", semantic.get("geometry_type", "")))
        if kind not in _GEOMETRY_TYPES:
            raise ValueError(f"unsupported resolved geometry type: {kind!r}")
        device = self.geometry.weight.device
        return self.output(torch.cat((self.geometry(torch.tensor([_GEOMETRY_TYPES.index(kind)], device=device)).squeeze(0), _fixed(_numbers(semantic), 64).to(device))).unsqueeze(0)).squeeze(0)


class BodyCandidateEncoder(CandidateEncoder):
    pointer_type = PointerType.BODY

    def __init__(self, face_dim: int = 128, metadata_dim: int = 16, output_dim: int = 128, pooling: str = "mean", use_relative_age: bool = True):
        super().__init__(output_dim=output_dim)
        if pooling not in {"mean", "mean-max", "max"}:
            raise ValueError("body pooling must be mean, mean-max, or max")
        self.pooling = pooling
        self.use_relative_age = use_relative_age
        self.face_projection = nn.Linear(64, face_dim)
        self.metadata_projection = nn.Linear(64, metadata_dim)
        pooled_dim = face_dim * (2 if pooling == "mean-max" else 1)
        self.projection = nn.Sequential(nn.Linear(pooled_dim + metadata_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, output_dim))

    @staticmethod
    def _face_rows(semantic: Any, metadata: Mapping[str, Any]) -> list[torch.Tensor]:
        rows = metadata.get("face_embeddings")
        if rows is None and isinstance(semantic, Mapping):
            rows = semantic.get("face_embeddings")
        if rows is not None and list(rows):
            return [_numbers(row) for row in rows]
        geometry = semantic.get("geometry") if isinstance(semantic, Mapping) else semantic
        if isinstance(geometry, Mapping) and geometry.get("faces"):
            return [_numbers(row) for row in geometry["faces"]]
        if hasattr(geometry, "Faces"):
            result = []
            for face in geometry.Faces():
                values = []
                for attr in ("Center", "Area"):
                    method = getattr(face, attr, None)
                    if callable(method):
                        value = method()
                        values.append(_numbers(value.toTuple() if hasattr(value, "toTuple") else value))
                if values:
                    result.append(torch.cat(values))
            if result:
                return result
        raise ValueError("BODY candidate requires actual body constituent geometry")

    def forward(self, semantic: Any, *, metadata: Mapping[str, Any] | None = None) -> torch.Tensor:
        metadata = dict(metadata or {})
        rows = torch.stack(self._face_rows(semantic, metadata))
        projected = self.face_projection(torch.stack([_fixed(row) for row in rows]).to(self.face_projection.weight.device))
        mean = projected.mean(0)
        pooled = torch.cat((mean, projected.max(0).values)) if self.pooling == "mean-max" else projected.max(0).values if self.pooling == "max" else mean
        typed_metadata = {"active": metadata.get("active", True), "predecessor_count": metadata.get("predecessor_count", 0), "relative_age": metadata.get("relative_age", 0) if self.use_relative_age else 0.0, "spatial": metadata.get("spatial", {})}
        meta = self.metadata_projection(_fixed(_numbers(typed_metadata)).to(self.metadata_projection.weight.device).unsqueeze(0)).squeeze(0)
        return self.projection(torch.cat((pooled, meta)).unsqueeze(0)).squeeze(0)


def default_encoders() -> dict[PointerType, nn.Module]:
    return {PointerType.FACE: FaceCandidateEncoder(), PointerType.EDGE: EdgeCandidateEncoder(), PointerType.BODY: BodyCandidateEncoder(), PointerType.PROFILE: ProfileCandidateEncoder(), PointerType.SKETCH_REFERENCE: SketchReferenceCandidateEncoder(), PointerType.RESOLVED_GEOMETRY: ResolvedGeometryCandidateEncoder()}


class RegistryContextEncoder(nn.Module):
    """Bounded typed registry summaries for ContextView."""

    def __init__(self, hidden_dim: int, mode: str = "NONE", output_dim: int | None = None):
        super().__init__()
        if mode not in {"NONE", "TYPE_POOLED"}:
            raise ValueError("registry context mode must be NONE or TYPE_POOLED")
        self.mode = mode
        self.output_dim = output_dim or hidden_dim
        self.projections = nn.ModuleDict({ptype.value: nn.Linear(128, self.output_dim) for ptype in PointerType})

    def forward(self, banks: Mapping[PointerType, Any]) -> torch.Tensor:
        if self.mode == "NONE":
            return torch.empty((0, self.output_dim))
        rows = [self.projections[ptype.value](bank.embeddings()).mean(0) for ptype in PointerType if (bank := banks.get(ptype)) is not None and len(bank)]
        return torch.stack(rows) if rows else torch.empty((0, self.output_dim))
