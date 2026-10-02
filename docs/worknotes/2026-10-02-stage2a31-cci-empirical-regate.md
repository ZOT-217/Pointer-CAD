# Stage2A.3.1 CCI Real-Model Empirical Re-gate

## Verdict

`STAGE2A31_READY_FOR_EXPANDED_9K`. The integrated path loads the fixed 80-record
natural corpus, uses `Construct the CAD model.` as X for every record, and runs
pretrained Qwen2.5-0.5B-Instruct with bf16 and rank-8 LoRA. Native B-Rep,
CandidateKey alignment, explicit causal targets, feedback, inference caching,
backward, and a bounded learnability check all passed. This gate tests
state/geometry-to-action learning under a constant task instruction; it does not
measure detailed language understanding.

## Frozen preparation and narrow repairs

The frozen manifest remained unchanged (SHA-256
`8f20e1e9e76a239fdd0fa40c5753e3393af2eda08345b9a57fd897a377e3d8d1`).
The `cadquery2crs` environment prepared 80/80 identities and 1,721 pre-action
sidecars without regenerating source CAD, membership, CRS, or supervision. The
preparer needed the frozen program's `_document` accessor. Native preparation
also needed a local fallback for undefined OCC UV normals and separately keyed
boundary/nonmanifold EDGE rows that the legacy face-adjacency graph omits.
Complete sidecars were reused on the final resumed pass and the authoritative
manifest was rewritten from the frozen corpus.

All prepared native arrays were finite: 109,451 FACE rows, 207,673 graph EDGE
rows, and 11,614 loose EDGE rows. The all-record audit mapped 32,371 grammar,
3,432 pointer, 945 scalar, and 662 structured-record targets. Every pointer GT
resolved in its causal JSON candidate bank; every FACE/EDGE bank key had a
matching owner-qualified native row, and all 207,673 graph EDGE rows had valid
adjacency. Every prepared X had the exact fixed instruction and adapter source.

The real forward exposed narrow Pointer-CAD integration faults: positionless
frozen structured records needed their unique typed action marker; prepared DGL
graphs and typed semantic tensors needed to move to the encoder device; bf16
Qwen hidden states needed conversion for float32 Stage2 heads; resolved
`PlaneSurface` needed a typed encoder entry. Native FACE/BODY gradients now
remain connected. These repairs were tested on the affected natural path.

## Real-model results

The visible GPU was an H100 80GB `3g.40gb` MIG instance with 42,412,802,048
bytes available. Qwen plus Stage2 had 14,929,312 trainable and 494,032,768
frozen parameters. LoRA targeted `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj`; UVNet and the Stage2 heads followed the
committed trainability configuration. AdamW used the default Stage2
`1e-4` learning rate.

Natural FACE, EDGE, BODY, PROFILE, SKETCH_REFERENCE, and RESOLVED_GEOMETRY
actions passed real forward/backward. Across natural actions, all four loss
components were finite. The 102-slot action gave `L_g=11.019`, `L_p=1.891`,
`L_s=5.113`; a FACE/record action gave `L_g=11.770`, `L_p=0.483`,
`L_r=7.952`. A zero-pointer Sketch action had finite grammar and FRAME3 record
loss, with pointer loss absent. LoRA, pointer projection/temperature, UVNet,
scalar and record heads, and registry/context modules all received nonzero
gradients on applicable actions, followed by optimizer steps.

A natural heterogeneous batch used EDGE slots in sample `00002:2` and
PROFILE/SKETCH_REFERENCE slots in `00008:4`. It carried 121 and 142 Qwen tokens,
four and two slots, and banks of 12 versus 2/1 candidates. Its loss and
backward were finite; all six slots resolved, isolated and batched top choices
agreed, and perturbing the second sample changed the first sample's pointer
logits by zero. Batch versus isolated bf16 logit differences were at most 0.092
on a 3.457 scale. The probe uses `forward_ragged` and the production loss heads
with explicit per-action target maps; `training_action_loss` itself accepts one
action.

The first natural pointer atom spanned three Qwen tokens. Feedback began at
`TokenSpan.end` (position 95, versus pointer start 92). Zeroing GT feedback
changed its own hidden/logits by zero and changed the second slot's hidden and
logits. Cached predicted-feedback inference selected the same four candidates
as the uncached reference; maximum bf16 logit difference was 0.0504, or 1.45%
of the observed logit scale, within the stated tolerance. The inference path
uses `past_key_values` and no teacher target.

Native parity checked 41 FACE and 65 graph EDGE rows plus six loose rows on a
natural action: 32x32x8 face features, 32x12 curve features, a 130-directed-edge
graph, and 128-dimensional pointer embeddings. Keyed Stage2 FACE/EDGE rows
matched a direct call to the original `UVNetEmbedder` module exactly under the
same weights. No legacy Pointer-CAD checkpoint was available for a checkpoint
ranking comparison. Stage2 owner keys, state/context, and its loose-edge
extension are intended differences; no unintended native regression was found.

## Cost and learnability

The low (1 slot, 141 tokens), medium (12 slots, 145 tokens), and high (102 slots,
422 tokens) actions each used exactly **one full Qwen forward**. Their measured
full action optimizer-step times were 0.640, 0.848, and 1.792 seconds. The high
action had 378 candidates per slot and peaked at 3.018 GB allocated / 3.246 GB
reserved, far below the 40 GB MIG limit. The old 102-slot path planned 103
full-prefix Qwen calls and OOMed. A one-pass trajectory uses one call per CAD
action: 19 for sample `00002`, 25 for `00013`, and 18 for `00472`. The cost
classification is `COST_READY_FOR_9K`. A rough serial 9K epoch extrapolation
from the medium action is 45.6 hours for about 193,612 actions; actual full
epoch throughput remains a 9K measurement.

On a fixed four-trajectory/eight-action natural subset, 12 epochs of bounded
micro-overfit reduced mean total loss from 25.463 to 1.820 (92.9%). Grammar,
pointer, scalar, and record means fell 83.5%, 79.5%, 99.2%, and 99.6%,
respectively. Pointer any-positive@1 rose from 0.182 to 0.364; grammar/value
accuracy rose from 0.120 to 0.526. Scalar loss fluctuated early, but the final
fixed-subset result shows a learnable signal without a hyperparameter search.

## Remaining scope

`BLOCK_BEFORE_9K`: none. `MEASURE_DURING_9K`: full-epoch throughput, long-action
distribution, and trained pointer behavior by candidate-bank size.
`DEFER_UNTIL_9K_RESULTS`: SplitFeature and historical EDGE natural coverage.
Historical EDGE needs real-model evidence before 100K. Neither absent case was
synthesized for this gate, and low early accuracy was not treated as a blocker.

The 19 small JSON files in
`docs/audits/2026-10-02-stage2a31-cci-empirical-regate/` contain the measured
inputs, loss and gradient values, time/VRAM observations, curves, decisions, and
limitations. Ignored sidecars and raw run outputs remain outside Git. Focused
CadQuery tests passed (5); Pointer-CAD causal/encoder tests available in the
CadQuery test environment passed (6, with the Transformers-only unit test
deselected). Real Qwen cache parity was exercised by the empirical runner in
the PointerCAD environment. Neither conda environment was modified.
