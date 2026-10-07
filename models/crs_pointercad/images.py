"""Model-independent, exact eight-view input manifest for Stage2."""
from __future__ import annotations

import json
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
        views = self.entries[tuple(identity)]
        paths = []
        for view in views:
            path = (self.root / view["locator"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"{tuple(identity)}: missing or escaping image view {view['view_index']}")
            with Image.open(path) as image:
                if image.size != (view["width"], view["height"]):
                    raise ValueError(f"{tuple(identity)}: image size mismatch at view {view['view_index']}")
            paths.append(path)
        return tuple(paths)
