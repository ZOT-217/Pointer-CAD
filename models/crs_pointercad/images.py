"""Model-independent, exact eight-view input manifest for Stage2."""
from __future__ import annotations

import json
import hashlib
import io
import os
from pathlib import Path
from typing import Sequence

from PIL import Image


class Stage2ImageManifest:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if manifest.get("format") != "stage2-eight-view-images-v1":
            raise ValueError("unsupported Stage2 image manifest format")
        self.entries = {}
        for entry in manifest["entries"]:
            identity = tuple(entry["corpus_identity"])
            if len(identity) != 4 or identity in self.entries:
                raise ValueError("duplicate or malformed image corpus identity")
            views = entry["views"]
            if len(views) != 8 or [view["view_index"] for view in views] != list(range(8)):
                raise ValueError(f"{identity}: exactly eight ordered views are required")
            for view in views:
                if view["source"] not in {"upstream_original", "canonical_render"}:
                    raise ValueError(f"{identity}: invalid image source")
                if view["source"] == "canonical_render" and not view.get("renderer_version"):
                    raise ValueError(f"{identity}: canonical image has no renderer version")
                if int(view["width"]) <= 0 or int(view["height"]) <= 0:
                    raise ValueError(f"{identity}: invalid image dimensions")
            self.entries[identity] = views

    def paths_for(self, identity: Sequence[str]) -> tuple[Path, ...]:
        key = tuple(identity)
        if key not in self.entries:
            raise ValueError(f"{key}: missing eight-view image manifest entry")
        views = self.entries[key]
        paths = []
        for view in views:
            path = (self.root / view["locator"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"{tuple(identity)}: missing or escaping image view {view['view_index']}")
            if "sha256" in view and hashlib.sha256(path.read_bytes()).hexdigest() != view["sha256"]:
                raise ValueError(f"{tuple(identity)}: image checksum mismatch at view {view['view_index']}")
            with Image.open(path) as image:
                if image.size != (view["width"], view["height"]):
                    raise ValueError(f"{tuple(identity)}: image size mismatch at view {view['view_index']}")
            paths.append(path)
        return tuple(paths)

    def images_for(self, identity: Sequence[str]) -> tuple[Image.Image, ...]:
        images = []
        for path in self.paths_for(identity):
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return tuple(images)


ARROW_FIELDS = tuple(f"image_{index}" for index in range(8))


class Zero2CADArrowImageProvider:
    """Read certified upstream image bytes through a per-process memory map.

    The index contains locators and provenance only. Dataset handles are opened
    lazily after worker fork/spawn and never included in pickled worker state.
    """

    def __init__(self, index: str | Path, dataset_root: str | Path):
        self.index = Path(index).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        document = json.loads(self.index.read_text(encoding="utf-8"))
        if document.get("format") != "stage2-zero2cad-arrow-images-v1":
            raise ValueError("unsupported Zero2CAD Arrow image index format")
        self.entries = {}
        for entry in document["entries"]:
            identity = tuple(entry["corpus_identity"])
            if len(identity) != 4 or identity in self.entries:
                raise ValueError("duplicate or malformed Arrow image corpus identity")
            if (entry.get("upstream_dataset") != "ADSKAILab/Zero-To-CAD-100k"
                    or entry.get("upstream_split") != "test"
                    or entry.get("source") != "upstream_original"
                    or entry.get("view_fields") != list(ARROW_FIELDS)
                    or entry.get("dimensions") != [256, 256]
                    or not isinstance(entry.get("upstream_row"), int)
                    or entry["upstream_row"] < 0):
                raise ValueError(f"{identity}: malformed eight-view Arrow locator")
            self.entries[identity] = entry
        self._dataset = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dataset"] = None
        state["_pid"] = None
        return state

    def _open(self):
        if self._dataset is None or self._pid != os.getpid():
            from datasets import load_from_disk
            split = self.dataset_root / "test"
            if not split.is_dir():
                raise ValueError(f"Zero2CAD Arrow test split is missing: {split}")
            dataset = load_from_disk(str(split))
            needed = set(ARROW_FIELDS) | {"cadquery_file", "num_renders"}
            missing = needed - set(dataset.column_names)
            if missing:
                raise ValueError(f"Zero2CAD Arrow split is missing fields: {sorted(missing)}")
            self._dataset = dataset.select_columns(["cadquery_file", "num_renders", *ARROW_FIELDS])
            self._pid = os.getpid()
        return self._dataset

    def images_for(self, identity: Sequence[str]) -> tuple[Image.Image, ...]:
        key = tuple(identity)
        if key not in self.entries:
            raise ValueError(f"{key}: missing eight-view Arrow image index entry")
        entry = self.entries[key]
        dataset = self._open()
        index = entry["upstream_row"]
        if index >= len(dataset):
            raise ValueError(f"{key}: upstream Arrow row {index} does not exist")
        row = dataset[index]
        if (row.get("num_renders") != 8 or
                hashlib.sha256(row.get("cadquery_file") or b"").hexdigest() != entry.get("cadquery_sha256")):
            raise ValueError(f"{key}: upstream Arrow row no longer matches certified CQ/eight-view source")
        images = []
        for field in ARROW_FIELDS:
            data = row.get(field)
            if not isinstance(data, bytes) or not data:
                raise ValueError(f"{key}: missing {field}")
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 256):
                        raise ValueError(f"{key}: invalid {field} format/mode/size")
                    images.append(image.copy())
            except Exception as exc:
                raise ValueError(f"{key}: cannot decode {field}: {exc}") from exc
        return tuple(images)
