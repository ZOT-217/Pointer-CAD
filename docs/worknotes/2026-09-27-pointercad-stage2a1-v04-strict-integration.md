# Pointer-CAD Stage 2A.1: v0.4 Strict Integration

## Result

Stage 2A interfaces were integrated into executable production paths. The
legacy `models.pointercad.PointerCAD` path remains selectable; expanded mode is
`models.crs_pointercad.CRSExpandedPointerCAD`.

## Main corrections

- `model_factory.py` now keeps `model.base_model` live and maps it to the real
  Qwen loader. An injected/tiny backbone is available only for dependency-light
  smoke tests.
- Expanded forward accepts normal multimodal `input_ids`, `attention_mask`,
  optional DGL B-Rep graphs, and causal `ExecutionState`; it no longer
  requires externally prepared hidden states.
- Active FACE/EDGE candidates consume native `UVNetEmbedder` pointer tensors.
  Historical candidates use the same 128-D encoder contract and fail closed if
  an owner+locator materializer has not supplied a native embedding.
- BODY embeddings use actual face embeddings or typed CadQuery face geometry,
  configurable pooling, real creation action indices, and a fully live age
  ablation.
- PROFILE, Curve3D, SketchReference, and ResolvedGeometry encoders preserve
  typed fields, loop/curve order, frames, roles, and NURBS control data.
- `RegistryContextEncoder(TYPE_POOLED)` is injected into the first decoder
  embedding; `NONE` is a real ablation.
- `grammar.py` now validates required fields, pointer types, list structure,
  Shell branch exclusivity, Chamfer TwoDistances EDGE->FACE records, and all
  twelve operation branches.
- Dynamic masks derive from state and decoder substate for incidence, owner,
  Boolean target/tool roles, profile consumers, and exclusions.
- Teacher-forced feedback uses the recorded target; inference feedback uses the
  selected target. Both update only later decoder embeddings.
- Runtime `BodyNode`/`ExecutionState` lineage now carries creation action index
  and creator operation rather than defaulting every body to action zero.

## Verification

The Pointer-CAD Stage 2A.1 test file passes `24` tests. It covers injected
Qwen forward, typed encoders, native FACE/EDGE bank space, body age ablation,
field grammar, dynamic masks, registry context wiring, teacher forcing,
inference selection, and finite backward.

The CRS-focused causal suite passes `37 passed, 2 skipped`. The Stage 2A.1
FIRST_1K runner is implemented at `tools/stage2a1_first1k.py`, but was not
executed in this local shell because the CadQuery environment lacks PyTorch and
the Pointer-CAD environment lacks CadQuery. The prior authoritative Stage 1
materializer remains 35,488 pointers, 17,569 banks, and zero failures; it is
not relabeled as a Stage 2.1 result.

## Boundary

The seven authoritative executor-tail cases remain explicitly
`KNOWN_EXECUTOR_TAIL_PENDING`. They do not block Stage 2A.1 model integration,
but must be fixed or quarantined before large-scale rollout. No model-quality,
pooling-choice, registry-fusion-choice, or numeric-head claim is made here.

See `docs/audits/2026-09-27-pointercad-stage2a1/v04_contract_implementation_matrix.json`
for the requirement-to-code-to-test status matrix.

POINTERCAD_STAGE2A1_V04_IMPLEMENTATION_READY_FOR_CCI_TRAINING
