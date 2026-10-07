#!/usr/bin/env python3
"""Bounded complete V1 state + upstream Arrow images + pretrained VLM smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoProcessor

from models.crs_pointercad import CRSExpandedPointerCAD, PreparedStage2Corpus, Stage2QwenCollator
from models.crs_pointercad.sequence import frozen_grammar_vocabulary
from models.crs_pointercad.training import training_action_loss


def run(*, frozen: Path, v1_native: Path, complete_inventory: Path,
        image_index: Path, arrow_root: Path, model_path: str, output: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("real pretrained VLM smoke requires CUDA")
    inventory = json.loads(complete_inventory.read_text(encoding="utf-8"))
    certified = [item for item in inventory["entries"] if item["classification"] == "COMPLETE_V1_RECORD"][:2]
    if len(certified) != 2:
        raise ValueError("need two individually certified complete V1 records")
    frozen_path = frozen / "manifest.json"
    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    if json.loads(image_index.read_text(encoding="utf-8"))["frozen_manifest_sha256"] != frozen_sha:
        raise ValueError("Arrow index is bound to another frozen corpus")
    document = json.loads(frozen_path.read_text(encoding="utf-8"))
    wanted = {tuple(item["identity"]): item for item in certified}
    frozen_entries = {}
    for entry in document["entries"]:
        identity = tuple(entry[k] for k in ("dataset", "sample_id", "source_variant_id", "approved_variant_id"))
        if identity in wanted:
            frozen_entries[identity] = entry
    del document
    if set(frozen_entries) != set(wanted):
        raise ValueError("certified V1 records are absent from frozen membership")
    with tempfile.TemporaryDirectory(prefix="stage2-arrow-v1-smoke-") as temporary:
        prepared_root = Path(temporary)
        prepared_entries = []
        for identity, item in wanted.items():
            source = v1_native / "records" / item["record_id"]
            destination = prepared_root / "records" / item["record_id"]
            shutil.copytree(source, destination)
            steps = []
            for action_index in range(item["expected_actions"]):
                step = destination / f"step-{action_index:06d}"
                required = ("state.json", "metadata.json", "face_features.npy",
                            "edge_features.npy", "loose_edge_features.npy")
                if not all((step / name).is_file() for name in required):
                    raise ValueError(f"{identity}: complete V1 inventory no longer matches step {action_index}")
                steps.append({"action_index": action_index, "path": step.relative_to(prepared_root).as_posix()})
            fixed = "Construct the CAD model."
            prepared_entries.append({
                "identity": list(identity), "steps": steps,
                "conditioning": {"text": fixed, "source": "expanded9k_fixed_constant",
                                 "sha256": hashlib.sha256(fixed.encode()).hexdigest()},
            })
        (prepared_root / "manifest.json").write_text(json.dumps({
            "format": "stage2a3-native-input-v1", "frozen_manifest_sha256": frozen_sha,
            "entries": prepared_entries,
        }) + "\n", encoding="utf-8")
        corpus = PreparedStage2Corpus(frozen, prepared_root, arrow_image_index=image_index,
                                      arrow_dataset_root=arrow_root)
        records = []
        for identity in wanted:
            supervision = json.loads((frozen / frozen_entries[identity]["supervision_artifact"]).read_text(encoding="utf-8"))
            pointer_actions = sorted({target["action_index"] for target in supervision["pointer_targets"]})
            if not pointer_actions:
                raise ValueError(f"{identity}: smoke needs one pointer-bearing action")
            record = {**supervision, "conditioning": corpus.conditioning_for(identity),
                      "images": corpus.images.images_for(identity)}
            records.append((identity, record, pointer_actions[0]))
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
        collator = Stage2QwenCollator(processor, grammar_vocabulary=frozen_grammar_vocabulary([r[1] for r in records]))
        sequences = [(identity, record, action_index, collator(record)) for identity, record, action_index in records]
        model = CRSExpandedPointerCAD(
            qwen_model=model_path, dtype="bf16", use_lora=True, lora_rank=8,
            hidden_dim=2048, pointer_dim=128, grammar_vocab_size=len(collator.grammar_vocabulary),
            registry_context_mode="TYPE_POOLED", use_native_brep=True,
        ).to("cuda")
        model.train()
        torch.cuda.reset_peak_memory_stats()
        measurements = []
        for identity, record, action_index, sequence in sequences:
            vision_before = model.vision_forward_count
            visual = model.encode_visual(sequence.pixel_values, sequence.image_grid_thw)
            state, brep = corpus.state_for(identity, action_index)
            torch.cuda.synchronize()
            began = time.perf_counter()
            loss, metrics = training_action_loss(model, sequence, record, action_index, state, brep,
                                                 visual_features=visual)
            torch.cuda.synchronize()
            forward = time.perf_counter() - began
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{identity}: nonfinite real pretrained loss")
            began = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward = time.perf_counter() - began
            measurements.append({
                "corpus_identity": identity, "action_index": action_index,
                "loss": float(loss.detach()),
                "parts": {key: float(metrics[key].detach()) for key in ("grammar", "pointer", "scalar", "record")},
                "forward_seconds": forward, "backward_seconds": backward,
                "visual_tokens": len(sequence.visual_token_positions),
                "vision_forwards": model.vision_forward_count - vision_before,
                "pointer_slots": metrics["pointer_stats"]["slots"],
            })
            model.zero_grad(set_to_none=True)
        result = {
            "status": "PASS", "model": model_path,
            "state_provider": "PreparedStage2Corpus.state_for over two individually complete V1 records",
            "image_provider": "Zero2CADArrowImageProvider", "records": measurements,
            "vision_forwards_per_trajectory": [item["vision_forwards"] for item in measurements],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "backward_succeeded": True,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen", "v1-native", "complete-inventory", "image-index", "arrow-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        frozen=args.frozen, v1_native=args.v1_native, complete_inventory=args.complete_inventory,
        image_index=args.image_index, arrow_root=args.arrow_root,
        model_path=args.model, output=args.output,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
