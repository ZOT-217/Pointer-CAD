# Pointer-CAD Stage 2A.2 Final Implementation Fixes

## Completed

- Added `CRSExpandedPointerCAD.forward_ragged()` as the loss-bearing training
  path. Each example owns its causal state, pointer slots, candidate banks,
  decoder substates, and teacher-forced target indices. The decoder is rerun
  after each GT feedback insertion, so later hidden states are causally changed
  while the producing pointer state is unchanged.
- Pointer BCE now accepts nested ragged slot lists and preserves v0.4 slot-first
  normalization. Zero-pointer examples remain valid batch members.
- Teacher-forced and inference decode now pass the substate for the current slot
  rather than the entire substate collection.
- Curve3D encoding now uses an explicit ordered typed-field stream plus GRU;
  NURBS control points/knots/weights and Polyline points are not silently
  truncated. Profile encoding uses ordered curve and loop GRUs while retaining
  outer/inner identity and frame data.
- Added trainable scalar and structured-record heads for POINT3, VECTOR3,
  DIRECTION3, AXIS3, PLANE3, and FRAME3. `ExpandedLoss` computes actual scalar
  and record losses, producing all four v0.4 components from model predictions.
- Added live config switches for ragged batching, autoregressive feedback, and
  numeric heads; expanded smoke/9K configs retain batch size greater than one.

## Verification

Targeted Stage 2A.2 tests pass: `32 passed`. They cover ragged batch counts and
bank sizes, zero-pointer coexistence, causal GT feedback, slot-local masks,
ordered Polyline/NURBS/Profile encodings, numeric prediction heads, four-part
loss, finite backward, and nonzero numeric gradients.

No FIRST_1K, checkpoint parity, CCI propagation experiment, or 9K training was
run in this task.

The seven Stage 1B.4 executor-tail cases remain unchanged and are still
`KNOWN_EXECUTOR_TAIL_PENDING`.

POINTERCAD_STAGE2A2_IMPLEMENTATION_READY_FOR_CCI_VALIDATION
