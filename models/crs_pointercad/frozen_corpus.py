"""Stage2 reader for approved, frozen CRS and causal supervision artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .candidates import CandidateView, ContextView
from .contracts import ExecutionState, PointerType


@dataclass(frozen=True)
class FrozenStage2Record:
    identity: tuple[str, str, str, str]
    program: Any
    supervision: dict[str, Any]


class FrozenStage2Corpus:
    """Read fixed membership and targets; no source recording or OCC enumeration."""

    def __init__(self, root: str | Path):
        from cadquery2crs.pipeline.approved_variants import ApprovedVariantManifest

        self.root = Path(root)
        self.manifest = ApprovedVariantManifest.load(self.root)

    def __len__(self) -> int:
        return len(self.manifest.entries)

    def __iter__(self) -> Iterator[FrozenStage2Record]:
        for entry in self.manifest.entries:
            program, supervision = self.manifest.load_frozen(self.root, entry)
            yield FrozenStage2Record(
                (entry["dataset"], entry["sample_id"], entry["source_variant_id"], entry["approved_variant_id"]),
                program, supervision,
            )

    @staticmethod
    def materialization_smoke(record: FrozenStage2Record) -> dict[str, int]:
        """Verify a frozen record reaches Stage2 prefix views and target banks."""
        trace = record.program.trace()
        supervision = record.supervision
        context_count = 0
        mapped = 0
        for target in supervision["pointer_targets"]:
            state = ExecutionState.from_runtime_snapshot(trace.at(target["snapshot_step"]))
            view = CandidateView(state)
            context = ContextView(state, view)
            if len(context.banks()) != len(PointerType):
                raise ValueError("Stage2 ContextView lacks a pointer type")
            context_count += 1
            pointer_type = PointerType("PTR_" + target["ref_kind"])
            bank = view.bank(pointer_type)
            expected = target["target_candidate"]
            # The model's candidate ordering is ephemeral. Compare external
            # identity fields, never the stored candidate-bank index.
            def same_key(entry):
                key = entry.external_key
                if pointer_type is PointerType.BODY:
                    return getattr(key, "value", key) == expected["owner"]
                if hasattr(key, "owner"):
                    return key.owner == expected["owner"] and key.local_index == expected["local_index"]
                return str(getattr(key, "value", key)) == expected["owner"]
            if sum(same_key(item) for item in bank) != 1:
                raise ValueError("frozen pointer target is not unique in Stage2 CandidateView")
            mapped += 1
        return {"context_views": context_count, "candidate_views": mapped,
                "pointer_targets": len(supervision["pointer_targets"]),
                "scalar_targets": len(supervision["parameter_targets"]),
                "structured_record_targets": len(supervision.get("structured_record_targets", []))}
