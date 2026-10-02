# Stage2A.3.1 Causal Input Closure (2026-09-30)

## Finding

Stage2A.3's loss-bearing sequence contained frozen `command.tokens` and the
current action slice, but no user text X. It therefore did not implement the
declared `P(A_t | S_t, X)` input contract. Its teacher feedback started at
`pointer_position + 1`, which crossed into the remaining subtokens when a
pointer atom had a multi-token Qwen span.

The original Pointer-CAD dataset reads `prompt_abs.txt` or `prompt_exp.txt`
and puts that prompt in the user turn through its chat template. The frozen
Zero2CAD admission path contains code, STEP, CRS, supervision and certification
but no per-record text annotation. The existing Pointer-CAD adapter for this
population used the fixed instruction `Construct the CAD model.`; this is the
only existing X available for those frozen identities. It is constant, so it
does not establish part-specific language conditioning quality.

## Repair

- The offline prepared-input manifest records each frozen identity's existing
  instruction, source and SHA-256. Explicitly proven dataset annotations take
  precedence when available; unknown or missing text fails closed.
- `PreparedStage2Corpus.collate_record` binds X to frozen supervision without
  changing the frozen corpus. `Stage2QwenCollator` renders the existing
  Pointer-CAD system/user chat template and maps command atoms after that
  prefix. `training_action_loss` feeds the full X prefix, current `S_t`, and
  only the current teacher-forced action command.
- Teacher forcing and cached/uncached inference consume explicit
  `TokenSpan.end` feedback positions. A three-token pointer atom stays free of
  its own feedback; later hidden states and pointer logits receive it.

## Verification and verdict

Local Pointer-CAD tests: 40 passed, including X versus X, future-target
isolation, multi-token span causality, tiny-Qwen cache parity and the existing
Stage2A tests. cadquery2crs focused tests: 10 passed. Changed Python files
compile and both diffs pass `git diff --check`.

`STAGE2A31_READY_FOR_CCI_REGATE` is the structural verdict. CCI must refresh
the deterministic prepared-input sidecar to carry X and repeat the empirical
checks; the 80-record frozen corpus and its membership remain unchanged.
