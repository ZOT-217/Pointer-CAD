"""Source bytes, frozen provenance, and all eight images form one mapping proof."""
from __future__ import annotations

import hashlib
import io
import json

import pytest
from PIL import Image

from tools.certify_stage2_arrow_images import certify


def test_certifier_emits_compact_index_only_after_exact_cq_and_eight_views(tmp_path):
    datasets = pytest.importorskip("datasets")
    frozen = tmp_path / "frozen"
    source = tmp_path / "source"
    dataset = tmp_path / "dataset"
    (source / "sources").mkdir(parents=True)
    frozen.mkdir()
    cq = b"import cadquery as cq\nresult = cq.Workplane().box(1, 2, 3)\n"
    (source / "sources" / "sample-00000.py").write_bytes(cq)
    (frozen / "manifest.json").write_text(json.dumps({"format": "stage2-clean-validation-v1", "entries": [{
        "dataset": "zero2cad100k-train", "sample_id": "sample-00000", "source_variant_id": "000",
        "approved_variant_id": "v000", "admission_status": "APPROVED",
        "provenance": {"source_sha256": hashlib.sha256(cq).hexdigest()},
    }]}), encoding="utf-8")
    source_manifest = frozen / "source_manifest.jsonl"
    source_manifest.write_text(json.dumps({"sample_index": 0, "source_path": "sources/sample-00000.py"}) + "\n")
    row = {"cadquery_file": [cq], "num_renders": [8]}
    for index in range(8):
        buffer = io.BytesIO()
        Image.new("RGB", (256, 256), (index * 20, 4, 5)).save(buffer, format="PNG")
        row[f"image_{index}"] = [buffer.getvalue()]
    datasets.Dataset.from_dict(row).save_to_disk(str(dataset / "test"))
    index_path = tmp_path / "index.json"
    audit = tmp_path / "audit"
    summary = certify(frozen_root=frozen, source_manifest=source_manifest, source_root=source,
                      dataset_root=dataset, index_path=index_path, audit_dir=audit, expected_count=1)
    assert summary["mapped_count"] == 1 and summary["failed_count"] == 0
    entry = json.loads(index_path.read_text())["entries"][0]
    assert entry["view_fields"] == [f"image_{i}" for i in range(8)]
    assert "locator" not in entry and "image_bytes" not in entry
    (source / "sources" / "sample-00000.py").write_bytes(cq + b"# changed\n")
    with pytest.raises(ValueError, match="image certification failed"):
        certify(frozen_root=frozen, source_manifest=source_manifest, source_root=source,
                dataset_root=dataset, index_path=tmp_path / "should_not_exist.json",
                audit_dir=tmp_path / "failed_audit", expected_count=1)
    failure = json.loads((tmp_path / "failed_audit" / "image_mapping_failures.json").read_text())
    assert failure["failed_count"] == 1
