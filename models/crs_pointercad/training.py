"""Loss-bearing action-level training from frozen Stage2 sequences."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .brep_bridge import FrozenCandidateKey, load_prepared_state, load_prepared_v2_state
from .contracts import ExternalKey, PointerType
from .heads import RECORD_DIMS
from .sequence import Stage2Sequence


_RECORD_FIELDS = {
    "POINT3": ("x", "y", "z"),
    "VECTOR3": ("x", "y", "z"),
    "DIRECTION3": ("x", "y", "z"),
    "AXIS3": ("origin", "direction", "length"),
    "PLANE3": ("origin", "normal", "x_axis"),
    "FRAME3": ("origin", "x_axis", "y_axis", "z_axis"),
}


def _record_values(record: Mapping[str, Any]) -> list[float]:
    kind = str(record["record_type"])
    fields = record["fields"]
    if kind not in _RECORD_FIELDS:
        raise ValueError(f"unsupported numeric record type {kind!r}")

    def flatten(value):
        if isinstance(value, Mapping):
            keys = ("x", "y", "z") if set(value) == {"x", "y", "z"} else tuple(sorted(value))
            return [number for key in keys for number in flatten(value[key])]
        if isinstance(value, (tuple, list)):
            return [number for item in value for number in flatten(item)]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("structured numeric target contains a nonnumeric field")
        return [float(value)]

    values = [number for field in _RECORD_FIELDS[kind] for number in flatten(fields[field])]
    if len(values) != RECORD_DIMS[kind]:
        raise ValueError(f"{kind} target has {len(values)} components; expected {RECORD_DIMS[kind]}")
    return values


def _target_key(value: Mapping[str, Any]):
    kind = str(value["kind"])
    if kind in {"FACE", "EDGE"}:
        return FrozenCandidateKey.from_dict(value)
    return ExternalKey(str(value["owner"]))


class PreparedStage2Corpus:
    """Index a prepared sidecar against the exact frozen manifest hash."""

    def __init__(self, frozen_root: str | Path, prepared_root: str | Path, *, native_backend: str = "v1",
                 migration_manifest: str | Path | None = None):
        self.frozen_root = Path(frozen_root).resolve()
        self.prepared_root = Path(prepared_root).resolve()
        self.migration_mode = migration_manifest is not None
        manifest_path = Path(migration_manifest) if self.migration_mode else self.prepared_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_format = f"stage2a3-native-{'migration' if self.migration_mode else 'input'}-{native_backend}"
        if native_backend not in {"v1", "v2"} or manifest.get("format") != expected_format:
            raise ValueError("unsupported prepared Stage2 geometry format")
        if self.migration_mode:
            if manifest.get("migration_mode") is not True or manifest.get("source_backend") != "v1_unreleased_individually_validated":
                raise ValueError("invalid unreleased migration provenance")
            if manifest.get("status") not in {"COMPLETE", "PARTIAL"}:
                raise ValueError("invalid unreleased migration status")
            expected = [tuple(item) for item in manifest.get("expected_identities", [])]
            entries = [tuple(item["identity"]) for item in manifest["entries"]]
            if len(set(expected)) != len(expected) or not set(entries).issubset(expected):
                raise ValueError("migration manifest entries differ from expected identities")
            if manifest.get("status") == "COMPLETE" and (set(entries) != set(expected) or manifest.get("failures")):
                raise ValueError("complete migration manifest has missing records or failures")
        elif manifest.get("migration_mode"):
            raise ValueError("unreleased migration data requires an explicit migration manifest")
        self.native_backend = native_backend
        frozen_sha = hashlib.sha256((self.frozen_root / "manifest.json").read_bytes()).hexdigest()
        if manifest["frozen_manifest_sha256"] != frozen_sha:
            raise ValueError("prepared geometry belongs to a different frozen manifest")
        self.entries = {tuple(item["identity"]): item for item in manifest["entries"]}
        if len(self.entries) != len(manifest["entries"]):
            raise ValueError("duplicate prepared Stage2 identity")

    def conditioning_for(self, identity: tuple[str, str, str, str]) -> dict[str, str]:
        """Get the pre-action X recorded by the offline preparation stage."""
        entry = self.entries[identity]
        conditioning = entry.get("conditioning")
        if not isinstance(conditioning, Mapping):
            raise ValueError("prepared Stage2 entry has no conditioning X")
        text = conditioning.get("text")
        source = conditioning.get("source")
        digest = conditioning.get("sha256")
        if not isinstance(text, str) or not text.strip() or source not in {
            "dataset_annotation", "legacy_pointercad_adapter_constant", "expanded9k_fixed_constant",
        } or digest != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise ValueError("prepared Stage2 conditioning provenance is invalid")
        return {"text": text, "source": source}

    def collate_record(self, record, collator):
        """Bind an approved frozen record to its immutable prepared X."""
        supervision = {**record.supervision, "conditioning": self.conditioning_for(record.identity)}
        return collator(supervision)

    def state_for(self, identity: tuple[str, str, str, str], action_index: int):
        entry = self.entries[identity]
        step = next(item for item in entry["steps"] if item["action_index"] == action_index)
        path = (self.prepared_root / step["path"]).resolve()
        if not path.is_relative_to(self.prepared_root):
            raise ValueError("prepared step path escapes sidecar root")
        if self.native_backend == "v2":
            return load_prepared_v2_state(path, path.parent)
        return load_prepared_state(path)


def training_action_loss(model, sequence: Stage2Sequence, supervision: Mapping[str, Any],
                         action_index: int, state, prepared_brep):
    """Run one causal action and score only its explicitly mapped targets."""
    action = next(item for item in sequence.action_boundaries if item.action_index == action_index)
    device = next(model.parameters()).device
    if sequence.conditioning_end <= 0 or action.start < sequence.conditioning_end:
        raise ValueError("action loss requires a nonempty conditioning X prefix")
    prefix = sequence.input_ids[:sequence.conditioning_end]
    ids = torch.cat((prefix, sequence.input_ids[action.start:action.end])).unsqueeze(0).to(device)
    mask = torch.ones_like(ids)

    def local(position: int) -> int:
        result = position - action.start + sequence.conditioning_end
        if not 0 <= result < ids.shape[1]:
            raise ValueError("target model position is outside its causal action")
        return result

    def feedback_local(position: int) -> int:
        result = position - action.start + sequence.conditioning_end
        if not sequence.conditioning_end <= result <= ids.shape[1]:
            raise ValueError("pointer feedback boundary is outside its causal action")
        return result

    pointer_targets = [(index, target) for index, target in enumerate(sequence.pointer_positions)
                       if target.action_index == action_index]
    specs = []
    for target_index, target in pointer_targets:
        key = target.target_candidate
        if key is None:
            raise ValueError("pointer target has no CandidateKey")
        pointer_type = PointerType("PTR_" + str(key["kind"]))
        source = next(item for item in supervision["pointer_targets"]
                      if item["action_index"] == action_index and item["position"] == target.command_position
                      and item["target_candidate"] == key)
        specs.append({
            "position": local(target.model_position),
            "feedback_position": feedback_local(sequence.feedback_positions[target_index]),
            "pointer_type": pointer_type,
            "target_key": _target_key(key),
            "decoder_substate": source.get("decoder_substate", {}),
        })
    output = model.forward_ragged(input_ids=ids, attention_mask=mask, states=[state],
                                  pointer_specs=[specs], breps=[prepared_brep] if prepared_brep is not None else None)

    grammar_indices = [index for index, target in enumerate(sequence.grammar_positions)
                       if target.action_index == action_index]
    grammar_logits = torch.stack([output.grammar_logits[0, local(sequence.grammar_positions[index].model_position)]
                                  for index in grammar_indices]) if grammar_indices else None
    grammar_targets = sequence.grammar_targets[grammar_indices].to(device) if grammar_indices else None

    positives = []
    for (_, target), bank in zip(pointer_targets, output.candidate_banks_by_example[0]):
        positives.append(bank.positive_mask((_target_key(target.target_candidate),), device=device))

    scalar_indices = [index for index, target in enumerate(sequence.scalar_positions)
                      if target.action_index == action_index]
    scalar_predictions = torch.stack([output.scalar_predictions[0, local(sequence.scalar_positions[index].model_position)]
                                      for index in scalar_indices]) if scalar_indices else None
    scalar_targets = torch.tensor([sequence.scalar_targets[index] for index in scalar_indices], device=device,
                                  dtype=scalar_predictions.dtype) if scalar_indices else None

    record_predictions = {}
    record_targets = {}
    for target, source in zip(sequence.structured_record_positions, sequence.structured_record_targets):
        if target.action_index != action_index:
            continue
        kind = str(source["record_type"])
        record_predictions.setdefault(kind, []).append(output.record_predictions[kind][0, local(target.model_position)])
        record_targets.setdefault(kind, []).append(_record_values(source))
    record_predictions = {key: torch.stack(values) for key, values in record_predictions.items()}
    record_targets = {key: torch.tensor(values, device=device, dtype=record_predictions[key].dtype)
                      for key, values in record_targets.items()}

    result = model.loss(grammar_logits=grammar_logits, grammar_targets=grammar_targets,
                      pointer_logits=output.pointer_logits_by_example[0], pointer_positive=positives,
                      scalar_predictions=scalar_predictions, scalar_targets=scalar_targets,
                      record_predictions=record_predictions, record_targets=record_targets)
    pointer_stats = {"slots": 0, "any_positive_at_1": 0, "top3": 0, "top5": 0,
                     "best_positive_rank_sum": 0, "by_type": {}, "by_bank_bucket": {}}
    for spec, logits, positive in zip(specs, output.pointer_logits_by_example[0], positives):
        if not logits.numel() or not positive.numel():
            continue
        positive = positive.to(device=logits.device, dtype=torch.bool)
        order = torch.argsort(logits, descending=True)
        positive_indices = torch.nonzero(positive, as_tuple=False).flatten()
        ranks = [(order == index).nonzero(as_tuple=False).item() + 1 for index in positive_indices]
        best = min(ranks)
        ptype = spec["pointer_type"].value.removeprefix("PTR_")
        size = logits.numel()
        bucket = "<=16" if size <= 16 else "17-32" if size <= 32 else "33-64" if size <= 64 else "65-128" if size <= 128 else ">128"
        pointer_stats["slots"] += 1; pointer_stats["any_positive_at_1"] += best == 1
        pointer_stats["top3"] += best <= 3; pointer_stats["top5"] += best <= 5; pointer_stats["best_positive_rank_sum"] += best
        for group, key in ((pointer_stats["by_type"], ptype), (pointer_stats["by_bank_bucket"], bucket)):
            item = group.setdefault(key, {"slots": 0, "any_positive_at_1": 0, "top3": 0, "top5": 0, "best_positive_rank_sum": 0})
            item["slots"] += 1; item["any_positive_at_1"] += best == 1; item["top3"] += best <= 3; item["top5"] += best <= 5; item["best_positive_rank_sum"] += best
    result[1]["pointer_stats"] = pointer_stats
    return result
