#!/usr/bin/env python3
"""Bounded real Qwen2.5-VL Stage2 smoke, cost, and four-trajectory learnability probe."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoProcessor

from models.crs_pointercad import CRSExpandedPointerCAD, PreparedStage2Corpus, Stage2QwenCollator
from models.crs_pointercad.sequence import frozen_grammar_vocabulary
from models.crs_pointercad.training import training_action_loss


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mean(values):
    return sum(values) / len(values) if values else 0.0


def load_rows(frozen: Path, prepared: PreparedStage2Corpus, count: int):
    manifest = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
    all_supervision = []
    rows = []
    for entry in manifest["entries"]:
        identity = tuple(entry[key] for key in ("dataset", "sample_id", "source_variant_id", "approved_variant_id"))
        supervision = json.loads((frozen / entry["supervision_artifact"]).read_text(encoding="utf-8"))
        all_supervision.append(supervision)
        if len(rows) < count and identity in prepared.entries and identity in prepared.images.entries:
            rows.append((identity, supervision))
    if len(rows) != count:
        raise ValueError(f"only {len(rows)} of {count} requested frozen image records are prepared")
    return rows, frozen_grammar_vocabulary(all_supervision)


def parts_dict(loss, parts):
    return {"total": float(loss.detach()), **{name: float(parts[name].detach())
            for name in ("grammar", "pointer", "scalar", "record")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-epochs", type=int, default=8)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20261007)
    prepared = PreparedStage2Corpus(args.frozen, args.prepared, args.images)
    rows, vocabulary = load_rows(args.frozen, prepared, 4)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    collator = Stage2QwenCollator(processor, grammar_vocabulary=vocabulary)
    sequences = {}
    action_data = []
    for identity, supervision in rows:
        record = {**supervision, "conditioning": prepared.conditioning_for(identity),
                  "image_paths": prepared.images.paths_for(identity)}
        sequence = collator(record)
        sequences[identity] = sequence
        for action in sequence.action_boundaries:
            if action.action_index in (1, 2):
                state, brep = prepared.state_for(identity, action.action_index)
                action_data.append((identity, record, sequence, action.action_index, state, brep))
    device = torch.device("cuda")
    model = CRSExpandedPointerCAD(qwen_model=str(args.model), dtype="bf16", use_lora=True,
                                  lora_rank=8, hidden_dim=2048, pointer_dim=128,
                                  grammar_vocab_size=len(vocabulary), registry_context_mode="TYPE_POOLED",
                                  use_native_brep=True).to(device)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    vision_start = model.vision_forward_count
    visual = {}
    vision_seconds = {}
    for identity, sequence in sequences.items():
        sync(); started = time.perf_counter()
        visual[identity] = model.encode_visual(sequence.pixel_values, sequence.image_grid_thw)
        sync(); vision_seconds[identity[1]] = time.perf_counter() - started
    vision_forwards = model.vision_forward_count - vision_start
    smoke = next(item for item in action_data if item[0][1] == "00002" and item[3] == 2)
    identity, record, sequence, index, state, brep = smoke
    original_forward = model.forward_ragged
    captured = {}
    def capture_forward(**kwargs):
        captured["kwargs"] = kwargs
        result = original_forward(**kwargs)
        captured["output"] = result
        return result
    model.forward_ragged = capture_forward
    sync(); started = time.perf_counter()
    loss, parts = training_action_loss(model, sequence, record, index, state, brep,
                                       visual_features=visual[identity])
    sync(); forward_seconds = time.perf_counter() - started
    if not torch.isfinite(loss):
        raise FloatingPointError("real multimodal smoke loss is not finite")
    sync(); started = time.perf_counter()
    loss.backward()
    sync(); backward_seconds = time.perf_counter() - started
    peak_allocated_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()
    model.forward_ragged = original_forward
    gradients = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = ("language_lora" if "lora_" in name else
                 "uvnet_gnn" if name.startswith("brep.") else
                 "feedback_adapter" if name.startswith("feedback.") else
                 "context_adapter" if name.startswith(("context_projection.", "brep_projection.", "registry_context.")) else
                 "typed_pointer_heads" if name.startswith(("query_heads.", "encoders.")) else
                 "grammar_head" if name.startswith("grammar_head.") else
                 "scalar_record_heads" if name.startswith("numeric_heads.") else "other")
        item = gradients.setdefault(group, {"parameters": 0, "nonzero_gradient_parameters": 0})
        item["parameters"] += parameter.numel()
        if parameter.grad is not None and torch.count_nonzero(parameter.grad).item():
            item["nonzero_gradient_parameters"] += parameter.numel()
    grid = sequence.image_grid_thw
    model.eval()
    with torch.no_grad():
        standard = original_forward(**captured["kwargs"])
        original_feedback = model.feedback_for
        try:
            model.feedback_for = lambda *unused: torch.zeros(model.hidden_dim, device=device)
            ablated = original_forward(**captured["kwargs"])
        finally:
            model.feedback_for = original_feedback
    model.train()
    pointer_positions = [slot["position"] for slot in captured["kwargs"]["pointer_specs"][0]]
    causal = {
        "pointer_positions": pointer_positions,
        "first_producing_hidden_max_delta": float((standard.hidden_states[0, pointer_positions[0]] -
                                                    ablated.hidden_states[0, pointer_positions[0]]).abs().max()),
        "later_pointer_hidden_max_delta": float((standard.hidden_states[0, pointer_positions[1]] -
                                                 ablated.hidden_states[0, pointer_positions[1]]).abs().max()),
        "candidate_bank_key_parity": all(
            [entry.external_key for entry in one] == [entry.external_key for entry in two]
            for one, two in zip(standard.candidate_banks_by_example[0],
                                ablated.candidate_banks_by_example[0])),
    }
    smoke_result = {
        "sample": identity[1], "action_index": index, "losses": parts_dict(loss, parts),
        "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "image_grid_thw": grid.tolist(), "visual_tokens_per_view":
            [(int(row[0] * row[1] * row[2]) // 4) for row in grid],
        "visual_tokens_total": len(sequence.visual_token_positions),
        "command_tokens": smoke[2].action_boundaries[index].end - smoke[2].action_boundaries[index].start,
        "multimodal_action_tokens": sequence.conditioning_end + smoke[2].action_boundaries[index].end - smoke[2].action_boundaries[index].start,
        "vision_forwards_for_four_trajectories": vision_forwards,
        "vision_forwards_per_trajectory": vision_forwards / len(sequences),
        "vision_forward_seconds_by_trajectory": vision_seconds,
        "gradients": gradients,
        "trainable": model.training_config(),
        "feedback_causality": causal,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "real_forward_backward.json").write_text(json.dumps(smoke_result, indent=2) + "\n")
    cost = {"sample": identity[1], "action_index": index,
            "source_image_resolution": [prepared.images.entries[identity][0]["width"],
                                        prepared.images.entries[identity][0]["height"]],
            "visual_tokens_per_view": smoke_result["visual_tokens_per_view"],
            "visual_tokens_total": smoke_result["visual_tokens_total"],
            "command_tokens": smoke_result["command_tokens"],
            "multimodal_action_tokens": smoke_result["multimodal_action_tokens"],
            "vision_forward_seconds": vision_seconds[identity[1]],
            "action_forward_seconds_with_reused_visuals": forward_seconds,
            "action_backward_seconds": backward_seconds,
            "action_forward_backward_seconds": forward_seconds + backward_seconds,
            "peak_allocated_bytes": smoke_result["peak_allocated_bytes"],
            "peak_reserved_bytes": smoke_result["peak_reserved_bytes"],
            "vision_forwards_per_trajectory": 1,
            "actions_in_trajectory": len(sequence.action_boundaries),
            "batch_size_sweep": False}
    (args.output / "cost.json").write_text(json.dumps(cost, indent=2) + "\n")
    print(json.dumps({"smoke": smoke_result["losses"], "forward_s": forward_seconds,
                      "backward_s": backward_seconds, "peak_gb": smoke_result["peak_allocated_bytes"] / 1e9}), flush=True)
    if args.smoke_only:
        return

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=0.01)
    def evaluate():
        model.eval()
        values = []
        with torch.no_grad():
            for identity, record, sequence, index, state, brep in action_data:
                value, metrics = training_action_loss(model, sequence, record, index, state, brep,
                                                      visual_features=visual[identity])
                values.append(parts_dict(value, metrics))
        model.train()
        return {key: mean([row[key] for row in values]) for key in values[0]}

    history = [{"epoch": 0, **evaluate()}]
    for epoch in range(1, args.micro_epochs + 1):
        for identity, record, sequence, index, state, brep in action_data:
            optimizer.zero_grad(set_to_none=True)
            value, _ = training_action_loss(model, sequence, record, index, state, brep,
                                            visual_features=visual[identity])
            if not torch.isfinite(value):
                raise FloatingPointError(f"micro-overfit loss nonfinite at epoch {epoch}")
            value.backward()
            optimizer.step()
        history.append({"epoch": epoch, **evaluate()})
        print(json.dumps(history[-1]), flush=True)
    outcome = {"trajectories": len(sequences), "actions": len(action_data),
               "epochs": args.micro_epochs, "vision_forwards_total": model.vision_forward_count,
               "history": history,
               "material_decrease": {key: history[-1][key] < 0.8 * history[0][key]
                                     for key in ("total", "pointer", "grammar", "scalar", "record")
                                     if history[0][key] > 0}}
    (args.output / "micro_overfit.json").write_text(json.dumps(outcome, indent=2) + "\n")


if __name__ == "__main__":
    main()
