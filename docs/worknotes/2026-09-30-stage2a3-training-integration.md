# Stage2A.3 Training Integration (2026-09-30)

## Scope

This change addresses the missing model-training integration identified by the
Stage2A.2 empirical gate. CRS representation, frozen corpus membership,
selection semantics and expanded 9K execution are unchanged.

## Implemented

- `Stage2QwenCollator` converts frozen command atoms into one explicit Qwen
  sequence and retains tokenizer spans, action boundaries, grammar/pointer/
  scalar/record positions, decoder substates and feedback insertion positions.
- `training_action_loss` uses those maps to slice a causal action and resolves
  GT candidates by `CandidateKey` in the prepared prefix bank.
- `forward_ragged` builds all teacher-forced GT feedback before one causal
  backbone call per example. Later slots see earlier feedback; a producing slot
  cannot see its own feedback.
- Inference pointer decoding uses Qwen `past_key_values` incrementally when the
  backbone supports caching. The tiny smoke backbone retains an uncached path.
- `PreparedBRep` and `prepare_native_snapshot` carry deterministic UV-Net face
  and edge tensors plus explicit owner-qualified CandidateKey row alignment.
  Ambiguous or locator-inconsistent matches fail closed.
- Expanded Qwen configuration now defaults to bf16 and baseline Pointer-CAD
  LoRA (rank 8, alpha 32, dropout 0.1, seven projection target groups).

## Verification boundary

The CadQuery environment passed the real prefix BRep bridge tests (2 passed).
Pointer-CAD source files pass Python compilation and diff checks. Real Qwen,
DGL, PEFT, natural frozen-corpus coverage, cache parity, native parity and
overfit remain CCI-only because this local shell does not contain the model
environment or the persistent corpus.

## Verdict

`STAGE2A3_TRAINING_INTEGRATION_HAS_BLOCKERS` locally. The implementation is
ready for a CCI empirical re-gate once the listed environment and artifact
requirements are available; no claim is made for expanded 9K readiness.

## Git Handoff

- Pointer-CAD START_HEAD: `15cbd32141240814ef6147b621ebf26278c79d2a`.
  It was fast-forward synchronized to `53aff6d4a8a3dfa983546f304849bbd8f85ce728`
  before implementation. END_HEAD: `82527db`; `git push origin main` succeeded.
- cadquery2crs START_HEAD: `2391361eee6fb6cbd3d48c5b0b078d9288e978b0`.
  It was fast-forward synchronized to `7286d7b4b2da0bb34e9818bc84a772033c9f87d5`
  before implementation. END_HEAD: `16d3cff`; `git push origin main` succeeded.
- The final fetch attempt returned a transient GitHub empty response; direct
  fast-forward pushes completed successfully after the local tests.
