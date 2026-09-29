# Stage2A.2 Real-Model Empirical Gate

## Decision

`STAGE2A2_EMPIRICAL_GATE_HAS_BLOCKERS`. The 80 frozen natural records load and a
real pretrained `Qwen/Qwen2.5-0.5B-Instruct` checkpoint runs forward, backward,
and an optimizer step on the H100 MIG slice. This is partial evidence: the real
Qwen step used one serialized natural command excerpt, an empty execution state,
and actual scalar/FRAME3 values placed at a diagnostic final position. It did
not exercise a valid Stage2 grammar or pointer target position.

## Evidence

- Both repositories were clean on `main`, fetched and fast-forward pulled.
  `cadquery2crs` started at `7286d7b4b2da0bb34e9818bc84a772033c9f87d5`;
  Pointer-CAD started at `92ae330c279bb27391d269fb1c3836db7f8db73b`.
- The frozen manifest SHA-256 is
  `8f20e1e9e76a239fdd0fa40c5753e3393af2eda08345b9a57fd897a377e3d8d1`.
  Its 80 `NATURAL_CLEAN` entries supplied 3,432 pointer, 945 scalar, and 662
  structured targets. A committed prefix of sample `00002` reconstructed a
  B-Rep with six faces and twelve edges. No CAD source was regenerated.
- Qwen loaded from the local configured checkpoint in float32. All 505,424,056
  Stage2 parameters were trainable; no PEFT/LoRA was active. The partial
  zero-pointer step had finite scalar and FRAME3 losses, nonzero Qwen and
  numeric-head gradients, peak allocated VRAM of 10.02 GB, and a 0.35 s
  measured step after checkpoint load. Grammar and pointer losses were absent.
- The model-side FACE/EDGE bank on a real frozen snapshot failed with
  `ValueError: native PTR_EDGE embedding is required`. The frozen loader's
  mapping smoke does not run UVNet. The CadQuery environment has the OCC
  snapshot but lacks the Qwen/DGL stack; the PointerCAD environment has the
  Qwen/DGL stack but lacks CadQuery. Neither environment was changed.
- A lower-bound Qwen rerun probe using natural action text and empty banks
  completed at one and twelve slots (two and thirteen backbone forwards).
  The 102-slot action from sample `00472` (334 Qwen tokens) failed inside the
  PyTorch CUDA allocator at 41.96 GB peak allocated VRAM before completing the
  forward. This probe omits UVNet,
  candidate scoring, and feedback, so it cannot certify full training cost.
  `BACKBONE_RERUN_IS_TRAINING_BLOCKER` on the available 40 GB MIG slice.

## Before Expanded 9K

`BLOCK_BEFORE_9K`: provide a production collator that preserves frozen command
token positions when forming Qwen inputs and places grammar, scalar, and record
targets at their causal positions. Provide the native B-Rep graph transfer and
candidate-key alignment required for real FACE/EDGE banks. Re-run natural
batch=2, zero-pointer, GT/predicted feedback, legacy FACE/EDGE parity, measured
full-cost, and four-component micro-overfit checks after that integration.

`MEASURE_DURING_9K`: candidate-bank-size accuracy and early model quality.
Untrained accuracy is not a release criterion. `DEFER_UNTIL_9K_RESULTS`:
SplitFeature and historical EDGE corpus exposure. Their absence alone does not
block the 9K diagnostic; historical EDGE should have real-model evidence before
100K expansion.

## Reproduction

Run `tools/stage2a2_empirical_probe.py corpus` with the `cadquery2crs`
interpreter, `LD_LIBRARY_PATH` pointing at that environment's `lib`, and
`PYTHONPATH` pointing at the repository's `src`. Run its `qwen` and `cost`
modes with the `PointerCAD` interpreter and this repository on `PYTHONPATH`.
The script reads the frozen corpus and local checkpoint only. Detailed status,
measurements, and unrun requirements are in
`docs/audits/2026-09-29-stage2a2-real-model-empirical-gate/`.
