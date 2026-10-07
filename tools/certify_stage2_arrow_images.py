#!/usr/bin/env python3
"""Certify frozen expanded-9K CQ identities and all eight upstream Arrow views.

Reads only frozen metadata, original CQ source bytes, and upstream Arrow. The
output index contains locators/provenance, never image bytes or new identities.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

from PIL import Image


FIELDS = [f"image_{index}" for index in range(8)]
DATASET = "ADSKAILab/Zero-To-CAD-100k"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_compact_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def source_manifest_rows(path: Path) -> dict[int, str]:
    rows = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        index = row["sample_index"]
        if index in rows:
            raise ValueError(f"duplicate source index {index}")
        rows[index] = row["source_path"]
    return rows


def certify(*, frozen_root: Path, source_manifest: Path, source_root: Path,
            dataset_root: Path, index_path: Path, audit_dir: Path,
            expected_count: int = 9087, limit: int | None = None) -> dict:
    from datasets import load_from_disk

    frozen_path = frozen_root / "manifest.json"
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    if frozen.get("format") != "stage2-clean-validation-v1":
        raise ValueError("unsupported frozen Stage2 manifest")
    frozen_entries = frozen["entries"]
    if limit is None and len(frozen_entries) != expected_count:
        raise ValueError(f"frozen manifest contains {len(frozen_entries)}, expected {expected_count}")
    if limit is not None:
        frozen_entries = frozen_entries[:limit]
    sources = source_manifest_rows(source_manifest)
    upstream = load_from_disk(str(dataset_root / "test"))
    needed = {"cadquery_file", "num_renders", *FIELDS}
    if needed - set(upstream.column_names):
        raise ValueError(f"upstream split is missing columns: {sorted(needed - set(upstream.column_names))}")
    upstream = upstream.select_columns(sorted(needed))
    source_root = source_root.resolve()
    mapped, failures, seen = [], [], set()
    for ordinal, entry in enumerate(frozen_entries):
        identity = [entry[k] for k in ("dataset", "sample_id", "source_variant_id", "approved_variant_id")]
        try:
            if tuple(identity) in seen:
                raise ValueError("duplicate frozen identity")
            seen.add(tuple(identity))
            if (identity[0] != "zero2cad100k-train" or identity[2:] != ["000", "v000"]
                    or entry.get("admission_status") != "APPROVED"):
                raise ValueError("unexpected frozen dataset/variant/admission")
            match = re.fullmatch(r"sample-(\d{5})", identity[1])
            if match is None:
                raise ValueError("unexpected sample ID")
            row_index = int(match.group(1))
            if row_index >= len(upstream):
                raise ValueError(f"upstream test row {row_index} is out of range")
            source_relative = sources.get(row_index)
            if source_relative != f"sources/{identity[1]}.py":
                raise ValueError(f"source manifest locator mismatch: {source_relative!r}")
            source_path = (source_root / source_relative).resolve()
            if not source_path.is_relative_to(source_root) or not source_path.is_file():
                raise ValueError("authoritative CQ source is missing or escapes source root")
            cq_bytes = source_path.read_bytes()
            upstream_row = upstream[row_index]
            if upstream_row["cadquery_file"] != cq_bytes:
                raise ValueError("authoritative CQ bytes differ from upstream cadquery_file")
            cq_sha = hashlib.sha256(cq_bytes).hexdigest()
            if entry.get("provenance", {}).get("source_sha256") != cq_sha:
                raise ValueError("frozen source provenance digest differs from CQ bytes")
            if upstream_row["num_renders"] != 8:
                raise ValueError(f"upstream num_renders={upstream_row['num_renders']}")
            for field in FIELDS:
                data = upstream_row.get(field)
                if not isinstance(data, bytes) or not data:
                    raise ValueError(f"missing {field}")
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                        if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 256):
                            raise ValueError(f"{field} has {image.format}/{image.mode}/{image.size}")
                except Exception as exc:
                    raise ValueError(f"{field} decode failed: {exc}") from exc
            mapped.append({"corpus_identity": identity, "upstream_dataset": DATASET,
                           "upstream_split": "test", "upstream_row": row_index,
                           "view_fields": FIELDS, "dimensions": [256, 256],
                           "source": "upstream_original", "cadquery_sha256": cq_sha})
        except Exception as exc:
            failures.append({"ordinal": ordinal, "corpus_identity": identity,
                             "error": f"{type(exc).__name__}: {exc}"})
    summary = {"format": "stage2-arrow-image-mapping-certification-v1",
               "expected_count": expected_count if limit is None else limit,
               "frozen_count": len(frozen_entries), "mapped_count": len(mapped),
               "failed_count": len(failures),
               "frozen_manifest_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
               "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
               "upstream_dataset": DATASET, "upstream_split": "test",
               "upstream_split_rows": len(upstream), "view_fields": FIELDS,
               "dimensions": [256, 256], "image_format": "RGB PNG",
               "method": "each frozen identity matched by exact CQ source bytes, frozen source digest, and eight decoded PNG views",
               "index": index_path.name}
    index_document = {"format": "stage2-zero2cad-arrow-images-v1",
                      "frozen_manifest_sha256": summary["frozen_manifest_sha256"],
                      "entries": mapped}
    if not failures and len(mapped) == len(frozen_entries):
        compact = json.dumps(index_document, sort_keys=True, separators=(",", ":")) + "\n"
        summary["index_sha256"] = hashlib.sha256(compact.encode("utf-8")).hexdigest()
    write_json(audit_dir / "image_mapping_9087.json", summary)
    write_json(audit_dir / "image_mapping_failures.json", {"failed_count": len(failures), "failures": failures})
    if failures or len(mapped) != len(frozen_entries):
        raise ValueError(f"image certification failed: {len(mapped)}/{len(frozen_entries)} mapped")
    write_compact_json(index_path, index_document)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=9087)
    args = parser.parse_args()
    print(json.dumps(certify(frozen_root=args.frozen, source_manifest=args.source_manifest,
                             source_root=args.source_root, dataset_root=args.dataset_root,
                             index_path=args.index, audit_dir=args.audit_dir,
                             expected_count=args.expected_count), sort_keys=True))


if __name__ == "__main__":
    main()
