#!/usr/bin/env python3
"""Copy a bounded, source-verified eight-view subset from local Zero-to-CAD Arrow.

This reads existing frozen membership and upstream Arrow; it never edits either.
The output is a derived image cache with a separate, model-independent manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

from PIL import Image


def source_row(sample_id: str) -> tuple[str, int]:
    if re.fullmatch(r"[0-9]{5}", sample_id):
        return "train", int(sample_id)
    if re.fullmatch(r"sample-[0-9]{5}", sample_id):
        return "test", int(sample_id.removeprefix("sample-"))
    raise ValueError(f"unknown Zero-to-CAD source identity {sample_id!r}")


def materialize(*, corpus: Path, dataset: Path, train_sources: Path,
                test_sources: Path, output: Path, limit: int,
                selected: set[str] | None = None) -> dict:
    from datasets import load_from_disk

    if limit < 1 or limit > 16:
        raise ValueError("bounded image probe requires 1 <= limit <= 16")
    frozen = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    rows = [entry for entry in frozen["entries"] if selected is None or entry["sample_id"] in selected]
    rows = rows[:limit]
    if not rows:
        raise ValueError("no frozen records selected")
    upstream = load_from_disk(str(dataset))
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in rows:
        identity = [entry[key] for key in ("dataset", "sample_id", "source_variant_id", "approved_variant_id")]
        if identity[2:] != ["000", "v000"]:
            raise ValueError(f"{identity}: upstream images require the original certified variant")
        split, index = source_row(identity[1])
        source = (train_sources / identity[1] / "source.py" if split == "train"
                  else test_sources / f"sample-{index:05d}.py")
        row = upstream[split][index]
        if not source.is_file() or source.read_bytes() != row["cadquery_file"]:
            raise ValueError(f"{identity}: frozen source does not match upstream Arrow row")
        if row["num_renders"] != 8:
            raise ValueError(f"{identity}: upstream row has {row['num_renders']} renders")
        destination = output / split / identity[1]
        destination.mkdir(parents=True, exist_ok=True)
        views = []
        for view_index in range(8):
            data = row[f"image_{view_index}"]
            if not isinstance(data, bytes) or not data:
                raise ValueError(f"{identity}: missing image_{view_index}")
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "PNG" or image.mode != "RGB":
                    raise ValueError(f"{identity}: image_{view_index} is not RGB PNG")
                width, height = image.size
            path = destination / f"view_{view_index}.png"
            path.write_bytes(data)
            views.append({"view_index": view_index, "locator": path.relative_to(output).as_posix(),
                          "width": width, "height": height, "source": "upstream_original",
                          "sha256": hashlib.sha256(data).hexdigest()})
        entries.append({"corpus_identity": identity, "upstream_split": split,
                        "upstream_row": index, "upstream_uuid": row["uuid"],
                        "cadquery_sha256": hashlib.sha256(row["cadquery_file"]).hexdigest(),
                        "views": views})
    manifest = {"format": "stage2-eight-view-images-v1", "entries": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-sources", type=Path, required=True)
    parser.add_argument("--test-sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--sample-id", action="append")
    args = parser.parse_args()
    result = materialize(corpus=args.corpus, dataset=args.dataset,
                         train_sources=args.train_sources, test_sources=args.test_sources,
                         output=args.output, limit=args.limit,
                         selected=set(args.sample_id) if args.sample_id else None)
    print(json.dumps({"records": len(result["entries"]), "manifest": str(args.output / "manifest.json")}))


if __name__ == "__main__":
    main()
