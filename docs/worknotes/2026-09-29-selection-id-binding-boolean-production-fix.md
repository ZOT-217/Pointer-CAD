# Selection Binding and Boolean Production Fix Handoff (2026-09-29)

## Git

Pointer-CAD branch: `main`, tracking `origin/main`.
START_HEAD and pulled remote HEAD:
`3b54a18e75c2234403d7c0abcd9c917f8c7905bf`.

Production runtime is maintained in the sibling cadquery2crs repository.
Its branch `main` was clean, fetched, and fast-forwarded before implementation.
Runtime START_HEAD and pulled remote HEAD:
`24243418dec4bc3b72eabea9b10a93de80cc514a`.
Runtime END_HEAD and pushed `origin/main`:
`7041c27c0ca68ba5b973506674f05090c3002b05`.
Runtime push succeeded: YES. This Pointer-CAD worknote and the requested audit
JSONs are committed separately in Pointer-CAD because the requested artifact
path belongs to this repository. Pointer-CAD END_HEAD and push receipt are
reported in the task completion message; a committed file cannot contain its
own commit hash.

## Contract

Opaque CRS IDs are variant-local, sequential first-use identifiers. The same
physical EDGE/FACE may receive a different numeric ID in a different legal
order variant. Within each variant, ID + immutable owner Body Node + stable
locator must identify exactly one physical entity. Candidate-bank indices are
ephemeral. The existing CRS validator, schema, and ID format are unchanged.
Action-local EDGE/FACE order is preserved; no global selection sorting exists.

The production binder rejects ambiguous owner-local locators before native
Fillet, Chamfer, Shell, and workplane operations. It also fixes a concrete FACE
workplane error: with two coincident extruded bodies, the selected face from
the second body previously bound to the first body's geometric role. It now
uses the unique actual physical owner.

## Hash Reassessment and Reproducibility

The 61 selection-related OLD/NEW CRS hash differences are
`UNVERIFIED_EQUIVALENT_VARIANT_CANDIDATE`. A changed value under the same
numeric ID across distinct order variants does not establish a correctness
regression. CCI stored diagnostics give the same source hash and feature-type
sequence with no material nonselection action differences, but do not prove
each pair has the same physical selected set, locator/owner binding, prefix
state, and STEP result. They are not certified equivalent either.

The dataset policy is a proposal: freeze approved variant manifests; use
order-preserving semantic trajectory digests for deduplication; record separate
STEP-backed equivalence decisions; admit unexpected OCC orders as explicit new
variant candidates with provenance instead of silently replacing a frozen
variant. This task changes neither the CRS ID contract nor the dataset variant
scheduler.

## Boolean 00135

Authoritative case: zero2cad100k-train, source SHA256
`31734a0507bea74e712bc025ebed80922847d4e227da9a7aa875bd3bcdd73611`,
CRS SHA256 `d6a41d5fd83e3737017de6ac7db4d96bd031d16fbc45dbfe9dd776fe36deb207`.
At `boolean_000006`, four prefix targets and no tools were unconditionally
mapped to zero NEW results. The handler now fuses actual targets and derives
output solids from the kernel, with GT output count and bbox validation-only.
The exact CCI source is unavailable locally; a four-target test covers the
audited semantic class and poisoned/no-GT execution. The local file named
00135 has a different SHA and was not used as the authoritative fixture.

## Verification and CCI Boundary

- CCI exception transport arrived at runtime START_HEAD; preserved unchanged.
  Local exception/stage tests: 5 passed. The prior CCI workers=32 result was
  460 success, 7 explicit runtime failure, 33 upstream failure, zero broken
  pools. It was not rerun here.
- Broad local affected regression: 239 passed, 67 subtests.
- Following the final FACE owner correction: 126 passed, 62 subtests in the
  affected recording/CRS/corpus subset.
- Fresh-process fixtures: 2-edge Fillet, 28-edge Chamfer, 4-edge Fillet control;
  9 independent processes total. Each variant's binding and causal pointer
  targets are checked; imported STEP shapes agree by bidirectional cut volume.
- Exact CCI 36-edge, exact 00135, frozen FIRST_500, workers=32, and all 61
  paired classifications remain CCI-only. No `PIPELINE_V2_READY_FOR_100K`
  claim is made.

Local gate: `SELECTION_BOOLEAN_FIX_READY_FOR_CCI_REGATE`.

## Engineering Handoff

- Recording call path: selected Workplane -> owner grouping ->
  `recording.selection_binding.validate_locator_bindings` -> native kernel
  operation -> ordered descriptors -> CRS `ConversionContext.intern_selection`.
- Replay path: variant-local ID and owner -> root selection locator ->
  `resolve_selection_use` -> one physical entity -> causal candidate bank and
  pointer target. Numeric candidate indices are temporary.
- Boolean call path: prefix body nodes -> `BooleanHandler.reconstruct` ->
  target fuse -> kernel solids -> cleanup -> count/geometry validation.
- Debug in the binder, `RecordingWorkplane.workplane`,
  `resolve_selection_use`, `ExecutionTrace.resolve_pointer_group`,
  and `BooleanHandler.reconstruct`.
- Trace: four 2x3x4 boxes at x=0,5,10,15, no tools -> union fuses four
  disjoint solids (96 mm^3) -> four output body nodes validate. The prior
  branch returned zero solids.
