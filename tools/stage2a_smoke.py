"""Run the dependency-light Stage 2A model plumbing smoke test."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.crs_pointercad import (
    BodyRecord, BodyVersionRegistry, CRSExpandedPointerCAD, ConstructionRecord,
    ConstructionRegistry, ExecutionState, ExternalKey, PointerType,
)


def main() -> None:
    torch.manual_seed(0)
    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=32)
    body = BodyRecord(ExternalKey("body_a"), 0, "Sketch", geometry={"faces": [{"center": [0.0, 0.0, 0.0], "area": 1.0}]}, spatial={"extent": [1.0, 2.0, 3.0]})
    profile = ConstructionRecord(ExternalKey("profile_a"), PointerType.PROFILE, 0, {"loops": [{"is_outer": True, "curves": [{"type": "Line3D", "start_point": [0, 0, 0], "end_point": [1, 0, 0]}]}]})
    reference = ConstructionRecord(ExternalKey("reference_a"), PointerType.SKETCH_REFERENCE, 0, {"type": "Line3D", "roles": ["feature_input"], "geometry": {"start_point": [0, 0, 0], "end_point": [1, 0, 0]}})
    resolved = ConstructionRecord(ExternalKey("resolved_a"), PointerType.RESOLVED_GEOMETRY, 0, {"type": "Plane3D", "origin": [0, 0, 0], "normal": [0, 0, 1]})
    class Snapshot:
        def candidates(self, kind, include_historical=False):
            return (SimpleNamespace(key="face_a", geometry=None, metadata={"historical": False, "native_embedding": torch.ones(128)}),) if str(kind).endswith("FACE") else (SimpleNamespace(key="edge_a", geometry=None, metadata={"historical": False, "native_embedding": torch.ones(128)}),) if str(kind).endswith("EDGE") else ()

    state = ExecutionState(
        active_brep_state=Snapshot(),
        action_index=1,
        body_version_registry=BodyVersionRegistry({body.key: body}),
        construction_registry=ConstructionRegistry({profile.key: profile, reference.key: reference, resolved.key: resolved}),
    )
    hidden = torch.randn(3, 16)
    slots = [PointerType.FACE, PointerType.EDGE, PointerType.BODY, PointerType.PROFILE,
             PointerType.SKETCH_REFERENCE, PointerType.RESOLVED_GEOMETRY]
    output = model(hidden, state, slots)
    targets = torch.tensor([1, 2, 3])
    pointer_positive = [torch.ones_like(logits, dtype=torch.bool) for logits in output.pointer_logits]
    loss, metrics = model.loss(grammar_logits=output.grammar_logits, grammar_targets=targets,
                               pointer_logits=output.pointer_logits, pointer_positive=pointer_positive)
    loss.backward()
    nonzero_gradients = sum(
        1 for parameter in model.parameters()
        if parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    )
    result = {
        "finite_grammar": bool(torch.isfinite(output.grammar_logits).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "valid_pointer_slots": metrics["valid_pointer_slots"],
        "backward": True,
        "nonzero_gradients": nonzero_gradients,
        "pointer_types": [slot.value for slot in slots],
    }
    destination = Path("docs/audits/2026-09-27-pointercad-stage2a/model_forward_smoke.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
