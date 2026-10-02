"""CRS-native semantic contracts used by the expanded Pointer-CAD model.

This module deliberately contains no CAD-kernel or tokenizer code.  It keeps
system identity (external keys) separate from model coordinates (embeddings
and ephemeral candidate indices) so the same contracts can be used by causal
materializers and training smoke tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Generic, Iterable, Mapping, TypeVar

import torch


class PointerType(str, Enum):
    FACE = "PTR_FACE"
    EDGE = "PTR_EDGE"
    BODY = "PTR_BODY"
    PROFILE = "PTR_PROFILE"
    SKETCH_REFERENCE = "PTR_SKETCH_REFERENCE"
    RESOLVED_GEOMETRY = "PTR_RESOLVED_GEOMETRY"


class ModelMode(str, Enum):
    LEGACY_POINTERCAD = "LEGACY_POINTERCAD"
    CRS_EXPANDED_POINTERCAD = "CRS_EXPANDED_POINTERCAD"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True, order=True)
class ExternalKey:
    """Opaque system key; it is never converted into model features."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("external key must be a non-empty string")


@dataclass(frozen=True)
class BodyRecord:
    key: ExternalKey
    creation_action_index: int
    creator_operation: str
    predecessor_keys: tuple[ExternalKey, ...] = ()
    active: bool = True
    geometry: Any = None
    spatial: Mapping[str, Any] = field(default_factory=dict)
    locators: Mapping[str, Any] = field(default_factory=dict)

    def age(self, action_index: int) -> int:
        return max(0, int(action_index) - self.creation_action_index)


@dataclass(frozen=True)
class ConstructionRecord:
    key: ExternalKey
    pointer_type: PointerType
    creation_action_index: int
    semantic: Any
    committed: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class BodyVersionRegistry:
    records: dict[ExternalKey, BodyRecord] = field(default_factory=dict)

    def add(self, record: BodyRecord) -> None:
        if record.key in self.records:
            raise ValueError(f"duplicate body key: {record.key.value}")
        self.records[record.key] = record

    def get(self, key: ExternalKey | str) -> BodyRecord | None:
        key = key if isinstance(key, ExternalKey) else ExternalKey(key)
        return self.records.get(key)

    def visible(self, *, active_only: bool = False) -> tuple[BodyRecord, ...]:
        rows = tuple(self.records.values())
        if active_only:
            rows = tuple(row for row in rows if row.active)
        return tuple(sorted(rows, key=lambda row: (row.creation_action_index, row.key.value)))


@dataclass
class ConstructionRegistry:
    records: dict[ExternalKey, ConstructionRecord] = field(default_factory=dict)

    def add(self, record: ConstructionRecord) -> None:
        if not record.committed:
            raise ValueError("only committed construction records enter the registry")
        if record.key in self.records:
            raise ValueError(f"duplicate construction key: {record.key.value}")
        self.records[record.key] = record

    def by_type(self, pointer_type: PointerType) -> tuple[ConstructionRecord, ...]:
        return tuple(sorted((r for r in self.records.values() if r.pointer_type is pointer_type),
                            key=lambda row: (row.creation_action_index, row.key.value)))


@dataclass
class ExecutionState:
    """Model-side view of one committed causal prefix."""

    active_brep_state: Any = None
    body_version_registry: BodyVersionRegistry = field(default_factory=BodyVersionRegistry)
    construction_registry: ConstructionRegistry = field(default_factory=ConstructionRegistry)
    action_index: int = 0
    # Native UV-Net candidate embeddings are a cache derived from the same
    # active prefix. They are optional; raw geometry remains authoritative.
    native_face_embeddings: Mapping[Any, torch.Tensor] = field(default_factory=dict)
    native_edge_embeddings: Mapping[Any, torch.Tensor] = field(default_factory=dict)
    native_body_face_embeddings: Mapping[Any, tuple[torch.Tensor, ...]] = field(default_factory=dict)
    historical_face_provider: Any = None
    historical_edge_provider: Any = None

    def clone(self) -> "ExecutionState":
        return ExecutionState(
            active_brep_state=self.active_brep_state,
            body_version_registry=BodyVersionRegistry(dict(self.body_version_registry.records)),
            construction_registry=ConstructionRegistry(dict(self.construction_registry.records)),
            action_index=self.action_index,
            native_face_embeddings=dict(self.native_face_embeddings),
            native_edge_embeddings=dict(self.native_edge_embeddings),
            native_body_face_embeddings=dict(self.native_body_face_embeddings),
            historical_face_provider=self.historical_face_provider,
            historical_edge_provider=self.historical_edge_provider,
        )

    def commit(self, *, bodies: Iterable[BodyRecord] = (), constructions: Iterable[ConstructionRecord] = ()) -> None:
        for body in bodies:
            self.body_version_registry.add(body)
        for record in constructions:
            self.construction_registry.add(record)
        self.action_index += 1

    @classmethod
    def from_runtime_snapshot(cls, snapshot: Any) -> "ExecutionState":
        """Adapt cadquery2crs ``ExecutionSnapshot`` without copying geometry."""
        bodies = BodyVersionRegistry()
        graph = getattr(snapshot, "body_graph", None)
        nodes = getattr(graph, "nodes", ()) if graph is not None else ()
        active = set(getattr(snapshot, "active_bodies", ()))
        raw = getattr(snapshot, "_bodies", {})
        body_creation = {}
        for node in nodes:
            key = ExternalKey(str(node.body_id))
            creation_index = getattr(node, "creation_action_index", None)
            if creation_index is None:
                raise ValueError("runtime snapshot is missing causal body creation index")
            creator_operation = getattr(node, "producer_operation", None)
            if creator_operation is None:
                raise ValueError("runtime snapshot is missing causal creator operation")
            bodies.records[key] = BodyRecord(
                key=key,
                creation_action_index=int(creation_index),
                creator_operation=str(creator_operation),
                predecessor_keys=tuple(ExternalKey(str(item)) for item in getattr(node, "predecessors", ())),
                active=node.body_id in active,
                geometry=raw.get(node.body_id),
            )
        # Preserve owner-local recovery metadata without making topology index
        # part of the model feature space.
        for pointer_name, kind_name in (("FACE", "FACE"), ("EDGE", "EDGE")):
            try:
                from cadquery2crs.representation import RefKind
                kind = RefKind[kind_name]
                for candidate in snapshot.candidates(kind, include_historical=True):
                    owner = getattr(candidate, "owner", None)
                    record = bodies.get(owner) if owner is not None else None
                    if record is not None:
                        locators = dict(record.locators)
                        locators[f"{pointer_name.lower()}:{getattr(candidate.key, 'local_index', 0)}"] = dict(getattr(candidate, "metadata", {})).get("locator")
                        bodies.records[record.key] = replace(record, locators=locators)
            except Exception:
                pass
        constructions = ConstructionRegistry()
        for key, profile in getattr(snapshot, "profiles", {}).items():
            constructions.records[ExternalKey(str(key))] = ConstructionRecord(
                ExternalKey(str(key)), PointerType.PROFILE, int(getattr(profile, "creation_action_index", 0)), getattr(profile, "geometry", profile)
            )
        for key, reference in getattr(snapshot, "sketch_references", {}).items():
            constructions.records[ExternalKey(str(key))] = ConstructionRecord(
                ExternalKey(str(key)), PointerType.SKETCH_REFERENCE, int(getattr(reference, "creation_action_index", 0)),
                getattr(reference, "record", reference)
            )
        for key, record in getattr(snapshot, "references", {}).items():
            if ExternalKey(str(key)) not in constructions.records:
                semantic = record.get("geometry", record) if isinstance(record, Mapping) else record
                constructions.records[ExternalKey(str(key))] = ConstructionRecord(
                    ExternalKey(str(key)), PointerType.RESOLVED_GEOMETRY, int(getattr(snapshot, "step_index", 0)), semantic
                )
        return cls(snapshot, bodies, constructions, int(getattr(snapshot, "step_index", 0)))


K = TypeVar("K")


@dataclass
class CandidateEntry(Generic[K]):
    external_key: K
    semantic: Any
    embedding: torch.Tensor | None = None
    legal_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.legal_metadata = MappingProxyType(dict(self.legal_metadata))
        if self.embedding is not None:
            if self.embedding.ndim != 1 or self.embedding.numel() != 128:
                raise ValueError("candidate embedding must be a flat 128-D tensor")
