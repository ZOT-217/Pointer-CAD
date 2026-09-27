"""Causal training materialization helpers and known executor exceptions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .contracts import ExecutionState, PointerType
from .candidates import CandidateView


KNOWN_EXECUTOR_TAIL = (
    {"sample_id": "sample-00045", "action_index": 16, "operation": "ShellFeature", "classification": "KNOWN_EXECUTOR_TAIL_PENDING"},
    {"sample_id": "sample-00081", "action_index": 3, "operation": "ShellFeature", "classification": "KNOWN_EXECUTOR_TAIL_PENDING"},
    {"sample_id": "sample-00322", "action_index": 13, "operation": "ShellFeature", "classification": "KNOWN_EXECUTOR_TAIL_PENDING"},
    {"sample_id": "sample-00564", "action_index": 8, "operation": "ShellFeature", "classification": "KNOWN_EXECUTOR_TAIL_PENDING"},
    {"sample_id": "sample-00656", "action_index": 3, "operation": "ShellFeature", "classification": "KNOWN_EXECUTOR_TAIL_PENDING"},
    {"sample_id": "sample-00900", "action_index": 6, "operation": "BooleanFeature", "classification": "KNOWN_EXECUTOR_TAIL_PENDING"},
    {"sample_id": "sample-00836", "action_index": None, "operation": "ShellFeature", "classification": "KNOWN_EXECUTOR_TAIL_PENDING", "failure_kind": "INVALID_BREP"},
)


@dataclass(frozen=True)
class MaterializationReport:
    pointer_positions: int
    candidate_banks: int
    model_materialization_failures: tuple[Mapping[str, Any], ...]
    known_executor_tail_failures: tuple[Mapping[str, Any], ...]
    unexpected_executor_failures: tuple[Mapping[str, Any], ...]

    @property
    def success(self) -> bool:
        return not self.model_materialization_failures and not self.unexpected_executor_failures


def materialize_causal_supervision(supervision: Any, trace: Any, *, model=None) -> dict[str, Any]:
    """Run a real cadquery2crs causal trace through Stage 2 registries/banks.

    This adapter intentionally resolves only committed prefix targets. Geometry
    encoders are optional for an inventory pass; supplying ``model`` additionally
    checks that every bank can produce its typed embedding.
    """
    pointer_positions = 0
    bank_keys = set()
    failures = []
    for target in getattr(supervision, "pointer_targets", ()):
        try:
            state = ExecutionState.from_runtime_snapshot(trace.at(target.snapshot_step))
            raw_kind = getattr(target.kind, "value", target.kind)
            pointer_type = PointerType(raw_kind if str(raw_kind).startswith("PTR_") else f"PTR_{raw_kind}")
            if model is None:
                bank = CandidateView(state).bank(pointer_type)
            else:
                bank = model.candidate_view(state, pointer_type)
            key = getattr(target, "target_candidate", None)
            bank.external_to_index(key)
            pointer_positions += 1
            bank_keys.add((target.snapshot_step, pointer_type.value))
        except Exception as exc:
            failures.append({"action_index": getattr(target, "action_index", None), "slot": getattr(target, "slot", None), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "pointer_positions": pointer_positions,
        "candidate_banks": len(bank_keys),
        "unresolved_targets": len(failures),
        "failures": failures,
        "model_path_checked": model is not None,
    }


def materialize_prefixes(samples: Iterable[Any], *, state_factory: Callable[[Any, int], ExecutionState], action_executor: Callable[..., Any] | None = None, candidate_view_factory: Callable[..., Any] | None = None) -> MaterializationReport:
    """Audit model inputs over committed prefixes without old sidecar banks.

    ``state_factory(sample, action_index)`` must return the state *before* that
    action.  An executor callback is optional for pure data dry-runs and, when
    present, receives only ``(state, action)``.
    """
    pointer_positions = 0
    bank_keys = set()
    model_failures = []
    known = []
    unexpected = []
    known_keys = {(row["sample_id"], row.get("action_index"), row["operation"]) for row in KNOWN_EXECUTOR_TAIL}
    for sample in samples:
        sample_id = getattr(sample, "sample_id", None) or sample.get("sample_id")
        actions = getattr(sample, "actions", None) or sample.get("actions", ())
        for action_index, action in enumerate(actions):
            try:
                state = state_factory(sample, action_index)
                targets = getattr(action, "pointer_targets", None) or action.get("pointer_targets", ()) if isinstance(action, Mapping) else getattr(action, "pointer_targets", ())
                for target in targets:
                    ptype = getattr(target, "pointer_type", None) or getattr(target, "kind", None)
                    ptype = PointerType(ptype.value if hasattr(ptype, "value") else ptype)
                    bank = candidate_view_factory(state, ptype) if candidate_view_factory else None
                    pointer_positions += 1
                    bank_key = (sample_id, action_index, ptype.value)
                    bank_keys.add(bank_key)
                    if bank is not None:
                        key = getattr(target, "external_key", None) or getattr(target, "target_candidate", None)
                        if key is not None:
                            bank.external_to_index(key)
            except Exception as exc:
                model_failures.append({"sample_id": sample_id, "action_index": action_index, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if action_executor is not None:
                try:
                    action_executor(state, action)
                except Exception as exc:
                    operation = getattr(action, "operation", None) or (action.get("operation") if isinstance(action, Mapping) else None)
                    operation = getattr(operation, "value", operation)
                    key = (sample_id, action_index, operation)
                    row = {"sample_id": sample_id, "action_index": action_index, "operation": operation, "error": f"{type(exc).__name__}: {exc}"}
                    if key in known_keys or any(sample_id == item["sample_id"] and (item.get("action_index") in {None, action_index}) and operation == item["operation"] for item in KNOWN_EXECUTOR_TAIL):
                        known.append(row)
                    else:
                        unexpected.append(row)
    return MaterializationReport(pointer_positions, len(bank_keys), tuple(model_failures), tuple(known), tuple(unexpected))
