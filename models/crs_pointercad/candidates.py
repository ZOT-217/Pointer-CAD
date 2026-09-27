"""Typed, prefix-causal candidate banks and deterministic views."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .contracts import (
    CandidateEntry,
    ConstructionRecord,
    ExecutionState,
    PointerType,
)


@dataclass(frozen=True)
class CandidateBank:
    pointer_type: PointerType
    entries: tuple[CandidateEntry[Any], ...]

    def __post_init__(self) -> None:
        keys = [entry.external_key for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("candidate external keys must be unique")

    @classmethod
    def build(cls, pointer_type: PointerType, entries: Iterable[CandidateEntry[Any]]) -> "CandidateBank":
        return cls(pointer_type, tuple(entries))

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def external_to_index(self, key: Any) -> int:
        for index, entry in enumerate(self.entries):
            if entry.external_key == key:
                return index
        raise KeyError(f"candidate key is not in {self.pointer_type.value}: {key!r}")

    def index_to_external(self, index: int) -> Any:
        if not 0 <= index < len(self.entries):
            raise IndexError(index)
        return self.entries[index].external_key

    def remap_targets(self, keys: Iterable[Any]) -> tuple[int, ...]:
        return tuple(self.external_to_index(key) for key in keys)

    def positive_mask(self, keys: Iterable[Any], *, device=None) -> torch.Tensor:
        positives = set(keys)
        return torch.tensor([entry.external_key in positives for entry in self.entries],
                            dtype=torch.bool, device=device)

    def embeddings(self, *, device=None, dtype=None) -> torch.Tensor:
        values = [entry.embedding for entry in self.entries]
        if not values:
            return torch.empty((0, 128), device=device, dtype=dtype or torch.float32)
        if any(value is None for value in values):
            raise ValueError("all candidate embeddings must be materialized before scoring")
        result = torch.stack(values)
        return result.to(device=device, dtype=dtype or result.dtype)

    def permute_storage(self, order: Sequence[int]) -> "CandidateBank":
        if sorted(order) != list(range(len(self.entries))):
            raise ValueError("order must be a permutation of candidate indices")
        return CandidateBank(self.pointer_type, tuple(self.entries[index] for index in order))


class CandidateView:
    """Build complete legal banks from a committed ``ExecutionState`` prefix."""

    def __init__(self, state: ExecutionState, *, encoders: Mapping[PointerType, Any] | None = None):
        self.state = state
        self.encoders = dict(encoders or {})

    def bank(self, pointer_type: PointerType | str, decoder_substate: Mapping[str, Any] | None = None) -> CandidateBank:
        pointer_type = pointer_type if isinstance(pointer_type, PointerType) else PointerType(pointer_type)
        entries = tuple(self._entries(pointer_type))
        substate = decoder_substate or {}
        entries = self._apply_masks(pointer_type, entries, substate)
        return CandidateBank.build(pointer_type, entries)

    def _apply_masks(self, pointer_type, entries, substate):
        """Derive legality from prefix state and decoded semantic substate."""
        def key_value(value):
            return getattr(value, "value", value)

        if not substate:
            return entries
        selected_edge = substate.get("selected_edge_key")
        owner = substate.get("owner_body_key")
        excluded = {key_value(value) for value in substate.get("excluded_keys", ())}
        role = substate.get("role")
        target_keys = {key_value(value) for value in substate.get("target_keys", ())}
        operation = substate.get("operation")
        selected_edge_key = key_value(selected_edge)
        result = []
        for entry in entries:
            metadata = entry.legal_metadata
            if key_value(entry.external_key) in excluded:
                continue
            if pointer_type is PointerType.BODY and role == "tool" and key_value(entry.external_key) in target_keys:
                continue
            allowed_operations = metadata.get("allowed_operations", metadata.get("consumers"))
            if pointer_type is PointerType.PROFILE and operation is not None and allowed_operations and operation not in set(allowed_operations):
                continue
            metadata_owner = key_value(metadata.get("owner"))
            if owner is not None and metadata_owner not in {None, key_value(owner)}:
                continue
            if selected_edge is not None and pointer_type is PointerType.FACE:
                incident = metadata.get("incident_edge_keys", metadata.get("incident_edges", ()))
                if incident and selected_edge_key not in {key_value(value) for value in incident}:
                    continue
                if metadata.get("edge_owner") not in {None, key_value(getattr(selected_edge, "owner", None))}:
                    if not incident:
                        continue
            result.append(entry)
        return tuple(result)

    def _entries(self, pointer_type: PointerType):
        state = self.state
        if pointer_type is PointerType.BODY:
            for record in state.body_version_registry.visible():
                semantic = {
                    "geometry": record.geometry,
                    "face_embeddings": state.native_body_face_embeddings.get(record.key, ()),
                }
                metadata = {
                    "historical": not record.active,
                    "active": record.active,
                    "spatial": dict(record.spatial),
                    "creator_operation_type": record.creator_operation,
                    "predecessor_count": len(record.predecessor_keys),
                    "relative_age": record.age(state.action_index),
                    "owner": record.key,
                }
                yield self._entry(record.key, semantic, metadata, pointer_type)
            return
        for record in state.construction_registry.by_type(pointer_type):
            yield self._entry(record.key, record.semantic, dict(record.metadata), pointer_type)
        if pointer_type in {PointerType.FACE, PointerType.EDGE}:
            snapshot = state.active_brep_state
            if snapshot is None or not hasattr(snapshot, "candidates"):
                return
            kind = {PointerType.FACE: "FACE", PointerType.EDGE: "EDGE"}[pointer_type]
            try:
                from cadquery2crs.representation import RefKind  # optional CRS runtime integration
                kind = RefKind[kind]
            except Exception:
                pass
            for candidate in snapshot.candidates(kind, include_historical=True):
                key = getattr(candidate, "key", candidate)
                semantic = getattr(candidate, "geometry", None)
                metadata = dict(getattr(candidate, "metadata", {}))
                if getattr(candidate, "owner", None) is not None:
                    metadata.setdefault("owner", candidate.owner)
                native = (state.native_face_embeddings if pointer_type is PointerType.FACE else state.native_edge_embeddings).get(key)
                if native is not None:
                    metadata["native_embedding"] = native
                yield self._entry(key, semantic, metadata, pointer_type)

    def _entry(self, key, semantic, metadata, pointer_type):
        encoder = self.encoders.get(pointer_type)
        embedding = encoder(semantic, metadata=metadata) if encoder is not None else None
        return CandidateEntry(key, semantic, embedding, metadata)

    def all_banks(self, decoder_substate: Mapping[str, Any] | None = None) -> dict[PointerType, CandidateBank]:
        return {pointer_type: self.bank(pointer_type, decoder_substate)
                for pointer_type in PointerType}


class ContextView:
    """Deterministic bounded context view over the same ExecutionState."""

    def __init__(self, state: ExecutionState, candidate_view: CandidateView, registry_encoder=None):
        self.state = state
        self.candidate_view = candidate_view
        self.registry_encoder = registry_encoder

    def banks(self) -> dict[PointerType, CandidateBank]:
        return self.candidate_view.all_banks()

    def features(self) -> torch.Tensor:
        banks = self.banks()
        if self.registry_encoder is None:
            return torch.empty((0, 0))
        return self.registry_encoder(banks)
