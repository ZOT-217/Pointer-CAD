"""Prepared native Pointer-CAD BRep inputs and explicit candidate alignment."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class FrozenCandidateKey:
    kind: str
    owner: str | None
    local_index: int

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> "FrozenCandidateKey":
        return cls(str(item["kind"]), item.get("owner"), int(item["local_index"]))


class PreparedSnapshot:
    """Pure-data stand-in for causal candidate enumeration in GPU training."""

    def __init__(self, metadata: Mapping[str, Any]):
        self.step_index = int(metadata["step_index"])
        self.active_bodies = tuple(metadata["active_bodies"])
        self.historical_bodies = tuple(metadata["historical_bodies"])
        self.body_graph = SimpleNamespace(nodes=tuple(SimpleNamespace(**item) for item in metadata["body_nodes"]))
        self._bodies = {item["body_id"]: item["geometry"] for item in metadata["body_nodes"]}
        self.profiles = {key: SimpleNamespace(**value) for key, value in metadata.get("profiles", {}).items()}
        self.sketch_references = {key: SimpleNamespace(**value) for key, value in metadata.get("sketch_references", {}).items()}
        self.references = dict(metadata.get("references", {}))
        self._candidates = {}
        for kind, rows in metadata["candidates"].items():
            self._candidates[kind] = tuple(
                SimpleNamespace(
                    key=FrozenCandidateKey.from_dict(item["key"]),
                    owner=item["key"].get("owner"),
                    geometry=item.get("geometry"),
                    metadata=item.get("metadata", {}),
                )
                for item in rows
            )

    def candidates(self, kind, *, body=None, include_historical=False):
        label = str(getattr(kind, "value", kind))
        rows = self._candidates.get(label, ())
        if body is not None:
            rows = tuple(item for item in rows if item.owner == body)
        if not include_historical:
            rows = tuple(item for item in rows if not item.metadata.get("historical"))
        return rows


@dataclass(frozen=True)
class PreparedBRep:
    """Deterministic UV-Net inputs for one causal prefix snapshot.

    ``face_keys`` and ``edge_keys`` are semantic CRS keys in graph row order;
    they are never inferred from an array index during candidate resolution.
    """

    graph: Any
    face_features: np.ndarray
    edge_features: np.ndarray
    face_keys: tuple[Any, ...]
    edge_keys: tuple[Any, ...]
    face_adjacency: tuple[tuple[int, int], ...]

    def alignment(self) -> dict[Any, tuple[str, int]]:
        result = {}
        for index, key in enumerate(self.face_keys):
            if key in result:
                raise ValueError("duplicate FACE CandidateKey in prepared BRep")
            result[key] = ("FACE", index)
        for index, key in enumerate(self.edge_keys):
            if key in result:
                raise ValueError("duplicate EDGE CandidateKey in prepared BRep")
            result[key] = ("EDGE", index)
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "face_count": len(self.face_keys),
            "edge_count": len(self.edge_keys),
            "face_keys": [_key_dict(key) for key in self.face_keys],
            "edge_keys": [_key_dict(key) for key in self.edge_keys],
            "face_adjacency": [list(pair) for pair in self.face_adjacency],
            "face_shape": list(self.face_features.shape),
            "edge_shape": list(self.edge_features.shape),
        }

    def to_dgl(self):
        """Create the official graph input without changing semantic keys."""
        try:
            import dgl
            import torch
        except ImportError as exc:
            raise RuntimeError("DGL and torch are required for native graph materialization") from exc
        src = [pair[0] for pair in self.face_adjacency]
        dst = [pair[1] for pair in self.face_adjacency]
        graph = dgl.graph((src, dst), num_nodes=len(self.face_keys))
        graph.ndata["x"] = torch.from_numpy(self.face_features)
        graph.edata["x"] = torch.from_numpy(self.edge_features)
        return dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True)


def _key_dict(key: Any) -> dict[str, Any]:
    kind = getattr(getattr(key, "kind", None), "value", getattr(key, "kind", None))
    return {"kind": kind, "owner": getattr(key, "owner", None), "local_index": getattr(key, "local_index", None)}


def validate_candidate_alignment(prepared: PreparedBRep, *, candidate_keys: Mapping[Any, Any] | None = None) -> dict[str, Any]:
    """Check one-to-one semantic key alignment before encoder execution."""
    alignment = prepared.alignment()
    if candidate_keys is not None:
        missing = [key for key in candidate_keys if key not in alignment]
        if missing:
            raise ValueError(f"candidate keys are absent from prepared native BRep: {missing[:3]!r}")
    if prepared.face_features.shape[0] != len(prepared.face_keys):
        raise ValueError("FACE tensor rows do not match FACE candidate keys")
    if prepared.edge_features.shape[0] != len(prepared.edge_keys):
        raise ValueError("EDGE tensor rows do not match EDGE candidate keys")
    return {"status": "PASS", "face_rows": len(prepared.face_keys), "edge_rows": len(prepared.edge_keys), "one_to_one": True}


def save_prepared_brep(prepared: PreparedBRep, root: str | Path) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "face_features.npy", prepared.face_features)
    np.save(root / "edge_features.npy", prepared.edge_features)
    (root / "metadata.json").write_text(json.dumps(prepared.metadata(), sort_keys=True, separators=(",", ":")), encoding="utf-8")


def load_prepared_brep(root: str | Path, *, graph: Any = None, key_loader=None) -> PreparedBRep:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if key_loader is None:
        key_loader = FrozenCandidateKey.from_dict
    face_keys = tuple(key_loader(item) for item in metadata["face_keys"])
    edge_keys = tuple(key_loader(item) for item in metadata["edge_keys"])
    prepared = PreparedBRep(
        graph,
        np.load(root / "face_features.npy", allow_pickle=False),
        np.load(root / "edge_features.npy", allow_pickle=False),
        face_keys,
        edge_keys,
        tuple(tuple(pair) for pair in metadata["face_adjacency"]),
    )
    validate_candidate_alignment(prepared)
    if graph is None:
        object.__setattr__(prepared, "graph", prepared.to_dgl())
    return prepared


def load_prepared_state(root: str | Path):
    """Load a causal step without importing CadQuery/OCC."""
    from .contracts import ExecutionState

    root = Path(root)
    metadata = json.loads((root / "state.json").read_text(encoding="utf-8"))
    snapshot = PreparedSnapshot(metadata)
    state = ExecutionState.from_runtime_snapshot(snapshot)
    brep = load_prepared_brep(root)
    return state, brep
