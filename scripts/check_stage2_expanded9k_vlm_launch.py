#!/usr/bin/env python3
"""Fail closed before an ACP VLM pilot; never builds a corpus or sidecars."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def check(corpus: Path, native: Path, image_index: Path, image_certification: Path,
          dataset_root: Path, model: str) -> dict:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor

    frozen_path = corpus / "manifest.json"
    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_entries = frozen.get("entries", [])
    frozen_ids = {tuple(row[k] for k in ("dataset", "sample_id", "source_variant_id", "approved_variant_id"))
                  for row in frozen_entries}
    if frozen.get("format") != "stage2-clean-validation-v1" or len(frozen_entries) != 9087 or len(frozen_ids) != 9087:
        raise ValueError("frozen expanded-9K release is not 9087 unique records")
    release = json.loads((corpus / "release_check.json").read_text(encoding="utf-8"))
    if (release.get("verdict") != "EXPANDED_9K_CORPUS_READY" or release.get("failures")
            or release.get("records") != 9087 or release.get("prepared_records") != 9087):
        raise ValueError("final Native V2 release_check has not passed for 9087 records")
    native_manifest = json.loads((native / "manifest.json").read_text(encoding="utf-8"))
    native_entries = native_manifest.get("entries", [])
    native_ids = {tuple(entry["identity"]) for entry in native_entries}
    if (native_manifest.get("format") != "stage2a3-native-input-v1"
            or native_manifest.get("frozen_manifest_sha256") != frozen_sha
            or len(native_entries) != 9087 or native_ids != frozen_ids):
        raise ValueError("Native V2 manifest is missing, incomplete, or bound to a different frozen release")
    index_bytes = image_index.read_bytes()
    certification = json.loads(image_certification.read_text(encoding="utf-8"))
    if (certification.get("mapped_count") != 9087 or certification.get("failed_count") != 0
            or certification.get("expected_count") != 9087
            or certification.get("frozen_manifest_sha256") != frozen_sha
            or certification.get("index_sha256") != hashlib.sha256(index_bytes).hexdigest()):
        raise ValueError("image mapping certification is missing or does not bind the compact index")
    image_manifest = json.loads(index_bytes)
    image_entries = image_manifest.get("entries", [])
    image_ids = {tuple(entry["corpus_identity"]) for entry in image_entries}
    fields = [f"image_{i}" for i in range(8)]
    if (image_manifest.get("format") != "stage2-zero2cad-arrow-images-v1"
            or image_manifest.get("frozen_manifest_sha256") != frozen_sha
            or len(image_entries) != 9087 or image_ids != frozen_ids
            or any(entry.get("view_fields") != fields or entry.get("dimensions") != [256, 256]
                   or entry.get("source") != "upstream_original" for entry in image_entries)):
        raise ValueError("certified 9087/9087 Arrow image index is missing or inconsistent")
    if not (dataset_root / "test" / "state.json").is_file():
        raise ValueError("Zero2CAD Arrow test split is unavailable")
    model_root = Path(model) if Path(model).is_dir() else Path(snapshot_download(model, local_files_only=True))
    config = AutoConfig.from_pretrained(str(model_root), local_files_only=True)
    if config.model_type != "qwen2_5_vl":
        raise ValueError("pilot model must be Qwen2.5-VL")
    AutoProcessor.from_pretrained(str(model_root), local_files_only=True, use_fast=False)
    weight_index = model_root / "model.safetensors.index.json"
    if not weight_index.is_file():
        raise ValueError("pretrained Qwen2.5-VL sharded weights are missing")
    shards = set(json.loads(weight_index.read_text(encoding="utf-8"))["weight_map"].values())
    if not shards or any(not (model_root / shard).is_file() or (model_root / shard).stat().st_size == 0 for shard in shards):
        raise ValueError("pretrained Qwen2.5-VL weight shards are incomplete")
    from safetensors import safe_open
    for shard in shards:
        with safe_open(model_root / shard, framework="pt", device="cpu") as weights:
            if not list(weights.keys()):
                raise ValueError(f"Qwen2.5-VL shard has no tensors: {shard}")
    return {"status": "PASS", "records": 9087, "image_views": 8,
            "frozen_manifest_sha256": frozen_sha,
            "image_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "model": str(model_root), "native": str(native)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--image-index", type=Path, required=True)
    parser.add_argument("--image-certification", type=Path, required=True)
    parser.add_argument("--arrow-dataset-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.corpus, args.native, args.image_index, args.image_certification,
                           args.arrow_dataset_root, args.model), sort_keys=True))


if __name__ == "__main__":
    main()
