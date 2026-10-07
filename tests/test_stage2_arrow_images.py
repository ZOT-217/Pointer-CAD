"""Arrow image provider ordering, failure, repeatability, and worker reopening."""
from __future__ import annotations

import io
import hashlib
import json
import os

import pytest
from PIL import Image

from models.crs_pointercad.images import Zero2CADArrowImageProvider


IDENTITY = ["zero2cad100k-train", "sample-00000", "000", "v000"]


def _png(index):
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), (index * 20, 7, 255 - index * 20)).save(buffer, format="PNG")
    return buffer.getvalue()


def _fixture(tmp_path, missing=None):
    datasets = pytest.importorskip("datasets")
    cq = b"original certified CQ bytes"
    rows = {f"image_{i}": [b"" if i == missing else _png(i)] for i in range(8)}
    rows.update({"cadquery_file": [cq], "num_renders": [8]})
    datasets.Dataset.from_dict(rows).save_to_disk(str(tmp_path / "test"))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"format": "stage2-zero2cad-arrow-images-v1", "entries": [{
        "corpus_identity": IDENTITY, "upstream_dataset": "ADSKAILab/Zero-To-CAD-100k",
        "upstream_split": "test", "upstream_row": 0,
        "view_fields": [f"image_{i}" for i in range(8)], "dimensions": [256, 256],
        "source": "upstream_original", "cadquery_sha256": hashlib.sha256(cq).hexdigest(),
    }]}), encoding="utf-8")
    return Zero2CADArrowImageProvider(index, tmp_path)


def test_arrow_provider_order_and_repeatability(tmp_path):
    provider = _fixture(tmp_path)
    first = provider.images_for(IDENTITY)
    second = provider.images_for(IDENTITY)
    assert len(first) == len(second) == 8
    assert [image.getpixel((0, 0))[0] for image in first] == list(range(0, 160, 20))
    assert [image.tobytes() for image in first] == [image.tobytes() for image in second]
    assert provider._pid == os.getpid()


def test_arrow_provider_missing_view_hard_failure(tmp_path):
    provider = _fixture(tmp_path, missing=4)
    with pytest.raises(ValueError, match="missing image_4"):
        provider.images_for(IDENTITY)


class _WorkerDataset:
    def __init__(self, provider):
        self.provider = provider

    def __len__(self):
        return 2

    def __getitem__(self, index):
        images = self.provider.images_for(IDENTITY)
        return os.getpid(), images[index].getpixel((0, 0))[0]


def test_arrow_provider_reopens_after_worker_fork(tmp_path):
    torch = pytest.importorskip("torch")
    provider = _fixture(tmp_path)
    provider.images_for(IDENTITY)
    parent_pid = provider._pid
    loader = torch.utils.data.DataLoader(_WorkerDataset(provider), batch_size=2,
                                          num_workers=1, multiprocessing_context="fork")
    worker_pid, colors = next(iter(loader))
    assert all(pid != parent_pid for pid in worker_pid.tolist())
    assert colors.tolist() == [0, 20]
