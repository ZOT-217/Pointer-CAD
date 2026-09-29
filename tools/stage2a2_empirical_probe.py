"""Read-only frozen-corpus and real-Qwen probes for the Stage2A.2 gate.

Run ``corpus`` in the cadquery2crs environment and ``qwen`` in PointerCAD.
Neither mode records CAD source or changes either environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time


CORPUS = Path("/mnt/afs/L202500475/cadquery2crs-corpus/stage2-clean-validation-v1")
CHECKPOINT = Path("/mnt/afs/L202500475/hf-data/models/Qwen2.5-0.5B-Instruct")


def corpus_probe() -> dict:
    from cadquery2crs.pipeline.approved_variants import ApprovedVariantManifest

    manifest = ApprovedVariantManifest.load(CORPUS)
    totals, types, sizes, action_slots, record_types = Counter(), Counter(), Counter(), Counter(), Counter()
    rows = []
    zero_actions = multi_actions = multi_positive = 0
    for entry in manifest.entries:
        program, supervision = manifest.load_frozen(CORPUS, entry)
        pointers = supervision["pointer_targets"]
        scalar = supervision["parameter_targets"]
        records = supervision["structured_record_targets"]
        tokens = supervision["command"]["tokens"]
        by_action = Counter(target["action_index"] for target in pointers)
        action_slots.update(by_action.values())
        record_types.update(target["record_type"] for target in records)
        zero_actions += sum(index not in by_action for index in range(len(supervision["command"]["action_boundaries"])))
        multi_actions += sum(count >= 2 for count in by_action.values())
        multi_positive += sum(len(target.get("positive_candidates", ())) >= 2 for target in pointers)
        totals.update(pointer=len(pointers), scalar=len(scalar), record=len(records))
        types.update(target["ref_kind"] for target in pointers)
        sizes.update(len(bank["candidates"]) for bank in supervision["candidate_banks"])
        rows.append({"sample_id": entry["sample_id"], "tokens": len(tokens),
                     "pointer_slots": len(pointers), "scalar_targets": len(scalar),
                     "record_targets": len(records)})
    # A trace is reconstructed from the committed CRS artifact, never source CAD.
    entry = next(item for item in manifest.entries if any(
        target["ref_kind"] in {"FACE", "EDGE"} for target in
        manifest.load_frozen(CORPUS, item)[1]["pointer_targets"]))
    program, supervision = manifest.load_frozen(CORPUS, entry)
    target = next(target for target in supervision["pointer_targets"] if target["ref_kind"] in {"FACE", "EDGE"})
    snapshot = program.trace().at(target["snapshot_step"])
    shape = snapshot.to_brep().shape
    buckets = {"<=16": 0, "17-32": 0, "33-64": 0, "65-128": 0, ">128": 0}
    for size, count in sizes.items():
        bucket = "<=16" if size <= 16 else "17-32" if size <= 32 else "33-64" if size <= 64 else "65-128" if size <= 128 else ">128"
        buckets[bucket] += count
    return {"frozen_loaded": len(rows), "partition": "NATURAL_CLEAN", "totals": dict(totals),
            "pointer_types": dict(types), "record_types": dict(record_types),
            "candidate_bank_size_buckets": buckets, "max_candidate_bank_size": max(sizes),
            "pointer_slots_per_action": dict(sorted(action_slots.items())),
            "zero_pointer_actions": zero_actions, "multi_pointer_actions": multi_actions,
            "multi_positive_targets_explicit": multi_positive,
            "real_brep_probe": {"sample_id": entry["sample_id"], "snapshot_step": target["snapshot_step"],
                                "faces": len(shape.Faces()) if shape is not None else 0,
                                "edges": len(shape.Edges()) if shape is not None else 0},
            "examples_by_pointer_count": {
                "low": min(rows, key=lambda row: row["pointer_slots"]),
                "medium": sorted(rows, key=lambda row: row["pointer_slots"])[len(rows) // 2],
                "high": max(rows, key=lambda row: row["pointer_slots"]),
            }}


def qwen_probe() -> dict:
    import torch
    from transformers import AutoTokenizer
    from models.crs_pointercad import CRSExpandedPointerCAD, ExecutionState

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    model = CRSExpandedPointerCAD(qwen_model=str(CHECKPOINT), grammar_vocab_size=1024,
                                   registry_context_mode="TYPE_POOLED").to("cuda")
    model.train()
    # The command text is natural frozen supervision, but this is not a Stage2
    # collator: Qwen token offsets do not equal frozen command-token offsets.
    first = next(json.loads(path.read_text()) for path in sorted(CORPUS.glob("NATURAL_CLEAN/*/*/*/supervision.json"))
                 if json.loads(path.read_text())["parameter_targets"] and
                 json.loads(path.read_text())["structured_record_targets"])
    text = " ".join(first["command"]["tokens"][:48])
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=96)["input_ids"].to("cuda")
    mask = torch.ones_like(input_ids)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.forward_ragged(input_ids=input_ids, attention_mask=mask,
                                  states=[ExecutionState()], pointer_specs=[[]])
    torch.cuda.synchronize()
    forward = time.perf_counter() - start
    # A real scalar/record value from the frozen record supervises one position.
    scalar = torch.full_like(output.scalar_predictions, float("nan"))
    scalar_target = first["parameter_targets"][0] if first["parameter_targets"] else None
    if scalar_target is not None:
        scalar[0, -1] = float(scalar_target["value"])
    record_targets = {}
    for target in first["structured_record_targets"]:
        if target["record_type"] == "FRAME3":
            fields = target["fields"]
            values = [fields[key][axis] for key in ("origin", "x_axis", "y_axis", "z_axis") for axis in ("x", "y", "z")]
            record_targets["FRAME3"] = torch.full_like(output.record_predictions["FRAME3"], float("nan"))
            record_targets["FRAME3"][0, -1] = torch.tensor(values, device="cuda")
            break
    loss, parts = model.training_loss(output, grammar_targets=None,
                                      scalar_targets=scalar, record_targets=record_targets)
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward = time.perf_counter() - start
    gradients = {}
    for prefix in ("base_model", "grammar_head", "numeric_heads.scalar", "numeric_heads.records"):
        gradients[prefix] = max((float(p.grad.norm()) for n, p in model.named_parameters()
                                 if n.startswith(prefix) and p.grad is not None), default=0.0)
    start = time.perf_counter()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    step = time.perf_counter() - start
    return {"scope": "partial_real_qwen_zero_pointer_probe_not_stage2_gate",
            "checkpoint": str(CHECKPOINT), "base_model_source": model.base_model_source,
            "dtype": str(next(model.base_model.parameters()).dtype), "device": str(input_ids.device),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "peft_lora": False, "input_qwen_tokens": int(input_ids.numel()),
            "loss": float(loss.detach()),
            "components": {key: float(parts[key].detach()) for key in ("grammar", "pointer", "scalar", "record")},
            "record_component_present": bool(record_targets), "scalar_component_present": scalar_target is not None,
            "gradients": gradients, "forward_seconds": forward, "backward_seconds": backward,
            "optimizer_seconds": step, "total_seconds": forward + backward + step,
            "peak_gpu_vram_bytes": torch.cuda.max_memory_allocated(),
            "backbone_forward_invocations": 1,
            "limitations": ["No frozen-token/Qwen-token position alignment", "No candidate bank or UVNet graph",
                            "Grammar and pointer losses absent, not zero-valued successes"]}


def cost_probe(slot_count: int) -> dict:
    """Measure a lower bound for the exact autoregressive Qwen rerun loop."""
    import torch
    from transformers import AutoTokenizer
    from models.crs_pointercad import CRSExpandedPointerCAD, ExecutionState, PointerType

    torch.manual_seed(0)
    selected = None
    for path in sorted(CORPUS.glob("NATURAL_CLEAN/*/*/*/supervision.json")):
        record = json.loads(path.read_text())
        by_action = Counter(target["action_index"] for target in record["pointer_targets"])
        matches = [index for index, count in by_action.items() if count == slot_count]
        if matches:
            action = matches[0]
            boundary = record["command"]["action_boundaries"][action]
            selected = (path, record, action, boundary)
            break
    if selected is None:
        raise ValueError(f"no natural action has {slot_count} pointer slots")
    path, record, action, boundary = selected
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    text = " ".join(record["command"]["tokens"][boundary["start"]:boundary["end"]])
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to("cuda")
    mask = torch.ones_like(input_ids)
    model = CRSExpandedPointerCAD(qwen_model=str(CHECKPOINT), grammar_vocab_size=1024,
                                   registry_context_mode="TYPE_POOLED").to("cuda").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    slots = [(min((index + 1) * input_ids.shape[1] // (slot_count + 1), input_ids.shape[1] - 1),
              PointerType.BODY) for index in range(slot_count)]
    torch.cuda.reset_peak_memory_stats()
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model.forward_ragged(input_ids=input_ids, attention_mask=mask,
                                      states=[ExecutionState()], pointer_specs=[slots])
        torch.cuda.synchronize()
        forward = time.perf_counter() - start
        scalar = torch.full_like(output.scalar_predictions, float("nan"))
        scalar[0, -1] = 0.0
        loss, _ = model.training_loss(output, grammar_targets=None, scalar_targets=scalar)
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward = time.perf_counter() - start
        start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step = time.perf_counter() - start
        return {"status": "COMPLETE", "sample_id": path.parts[-3], "action_index": action,
                "frozen_command_tokens": boundary["end"] - boundary["start"],
                "qwen_tokens": int(input_ids.numel()), "pointer_slots": slot_count,
                "candidate_bank_sizes": [0] * slot_count, "backbone_forwards": slot_count + 1,
                "forward_seconds": forward, "backward_seconds": backward,
                "optimizer_seconds": step, "total_seconds": forward + backward + step,
                "peak_gpu_vram_bytes": torch.cuda.max_memory_allocated(),
                "scope": "Qwen rerun lower bound only; empty banks, no feedback, no BRep"}
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        return {"status": "CUDA_FAILURE", "sample_id": path.parts[-3], "action_index": action,
                "frozen_command_tokens": boundary["end"] - boundary["start"],
                "qwen_tokens": int(input_ids.numel()), "pointer_slots": slot_count,
                "candidate_bank_sizes": [0] * slot_count,
                "peak_gpu_vram_bytes": torch.cuda.max_memory_allocated(),
                "error": str(exc).splitlines()[0],
                "scope": "Qwen rerun lower bound only; empty banks, no feedback, no BRep"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("corpus", "qwen", "cost"))
    parser.add_argument("--slots", type=int, choices=(1, 12, 102))
    args = parser.parse_args()
    if args.mode == "cost" and args.slots is None:
        parser.error("cost requires --slots")
    result = corpus_probe() if args.mode == "corpus" else qwen_probe() if args.mode == "qwen" else cost_probe(args.slots)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
