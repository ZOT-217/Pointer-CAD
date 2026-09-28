# Pointer-CAD Stage 2A Implementation Map

The live legacy path remains available under `models.pointercad.PointerCAD`.
The expanded path is isolated in `models/crs_pointercad/` and is selected by
`models.model_factory.from_config()` or `build_pointercad()`.

| Responsibility | Existing implementation | Stage 2A boundary |
|---|---|---|
| Dataset loading/materialization | `dataset/dataset.py`, `preprocessing/json2vec.py` | `crs_pointercad/materialization.py` records prefix state, banks, and known executor exceptions |
| Command serialization | `cadmodel/model.py`, `misc.py` | `crs_pointercad/grammar.py::ActionAST` validates typed semantic actions before CRS execution |
| Current B-Rep | `models/brep_embed.py` (`UVNetEmbedder`) | `FaceCandidateEncoder`/`EdgeCandidateEncoder` preserve 128-D geometry interface; legacy tensors remain untouched |
| LLM input | `models/processor.py` and `PointerCAD.forward()` | `RegistryContextEncoder` exposes `NONE`/`TYPE_POOLED` hook; expanded model accepts decoder hidden states |
| Decoder/value head | `models/pointercad.py` | `CRSExpandedPointerCAD.grammar_head` is a typed grammar hook; old Qwen decoder remains default |
| Pointer head/scoring | legacy `pointer_head`, `pointer_tau` | `TypedPointerHeads` gives per-type cosine queries/scales and `PointerFeedback` |
| Pointer loss | `metrics/criterion.py` | `pointer_bce_per_slot()` and `ExpandedLoss` implement multi-positive per-slot normalization and zero-slot safety |
| Autoregressive decode | `PointerCAD.predict()` | `CRSExpandedPointerCAD.execute_decoded()` accepts validated AST and calls a causal executor |
| Evaluation | `test.py`, `metrics/*` | `MaterializationReport` separates model/materialization failures, known executor-tail cases, and unexpected failures |

## Runtime contract

`ExecutionState` contains the committed prefix (`active_brep_state`, body
version registry, construction registry, action index). `CandidateView` builds
typed banks from that state only. External keys are system-side dispatch
metadata; candidate indices are generated per bank and never embedded.

The known Stage 1B.4 executor tail is machine-readable in
`crs_pointercad.materialization.KNOWN_EXECUTOR_TAIL` and is reported as
`KNOWN_EXECUTOR_TAIL_PENDING`.

