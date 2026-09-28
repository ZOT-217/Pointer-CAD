# Pointer-CAD Stage 2A Production Implementation

## Scope

Stage 2A adds CRS-native model plumbing while preserving the legacy B0 model.
No optimizer run, CCI experiment, or full training was launched.

## Architecture-to-code mapping

- `ExecutionState`, `BodyVersionRegistry`, and `ConstructionRegistry` are in `models/crs_pointercad/contracts.py`.
- `CandidateBank` and prefix-only `CandidateView` are in `candidates.py`.
- Typed 128-D encoders and registry context hooks are in `encoders.py`.
- `ActionAST` and operation grammar are in `grammar.py`.
- Typed query heads, feedback timing, and multi-positive loss are in `heads.py`.
- `CRSExpandedPointerCAD` is the backbone-agnostic expanded model shell in `model.py`.
- Prefix materialization accounting and the seven non-blocking Stage 1B.4 exceptions are in `materialization.py`.

## Legacy compatibility

`models.pointercad.PointerCAD` remains the default and now explicitly reports
`LEGACY_POINTERCAD`. `config/legacy_b0.yaml` retains the old restricted
Extrude/Chamfer/Fillet FACE/EDGE task. Expanded code is selected only through
`mode: CRS_EXPANDED_POINTERCAD` and does not alter the old forward path.

## Candidate and encoder behavior

Banks are keyed by opaque external system keys but expose only semantic records
and 128-D embeddings to model heads. Active and historical FACE/EDGE use one
typed bank each. BODY, PROFILE, SKETCH_REFERENCE, and RESOLVED_GEOMETRY have
replaceable baseline encoders. BODY pooling is configurable (`mean`/`max`) and
relative age is a toggle.

## Decoder, feedback, and loss

Decoder output becomes a validated `ActionAST` before executor invocation.
Feedback is `type_embedding + W_feedback[type](selected_embedding)` and is
added only to positions after the pointer slot. Teacher forcing can pass the
recorded candidate embedding; inference passes the selected candidate embedding.
Pointer BCE averages candidates within each slot, then slots, and supports
multiple positives. Empty pointer batches return finite zero loss.

## Verification

Local smoke coverage is recorded in the Stage 2A audit JSON directory. The
authoritative Stage 1B.4 materialization remains 35,488 pointer positions,
17,569 banks, and zero materialization failures. The seven known executor-tail
cases remain pending and are not reclassified as Stage 2A model blockers.

## CCI-only decisions

Body pooling quality, registry context winner, continuous record heads, and
final loss weights remain experiments. Learned memory, retrieval, RL, and
large-scale training are out of scope.

