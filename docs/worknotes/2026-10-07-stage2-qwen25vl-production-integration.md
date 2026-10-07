# Stage2 Qwen2.5-VL production integration (2026-10-07)

START_HEAD: `74e3c09758f3bb4c21b8ab7736cf42181c436c32`. Work was done on
`stage2-qwen25vl-production-integration` in an isolated local clone. The source
checkout could not create an AFS Git ref (`ENOSPC`); a later SSH fetch used
GitHub host keys cross-checked against the GitHub HTTPS metadata API. The
running ACP job, native sidecars, frozen membership, Pipeline V2 and separate
Native V2 branches were not edited.

## Contract and implementation

The intended Stage2 input is current causal `S_t`, eight ordered final-shape
RGB images and the exact instruction `Construct the CAD model.`. Text-only
conditioning remains an explicit debug/ablation mode. The production
configuration now names `Qwen/Qwen2.5-VL-3B-Instruct`, bf16, language LoRA,
frozen base language weights, frozen vision tower and frozen visual merger.
Stage2 UVNet/GNN, context/feedback adapters and typed losses remain trainable.

The official `Qwen2_5_VLProcessor` renders the content-list user turn. Its
image processor receives eight real PIL images and emits `pixel_values` and
`image_grid_thw`; the tokenizer's expanded image tokens precede every frozen
command atom. Stage2 maps each command atom to a complete final-token span,
preserves target/feedback maps, and scores only the current action. A single
constant assistant-side token after the images owns the current-state context
injection. The official Qwen visual tower/merger produces the visual features;
the official `get_rope_index` produces MRoPE positions; then the Qwen decoder
receives Stage2's context and post-span feedback additions. Stage2 checks that
visual feature/token counts match and that context never occupies an image
token. `TokenSpan.end` remains the first feedback position. Frozen visual
features can be reused in memory across actions of one trajectory.

The `PreparedStage2Corpus.state_for` native V1 boundary is unchanged. The
separate Native V2 contract selects storage below that same API and requires
V1/V2 action parity; this branch does not select or rewrite either backend.
Images live in an independent derived manifest keyed by the existing four-part
corpus identity. Exactly views 0 through 7 must be present in order.

## Upstream image proof

The local Arrow dataset at
`/mnt/afs/L202500475/hf-data/datasets/Zero-To-CAD-100k` retains eight `image_*`
binary fields even though the materialized CQ/STEP source trees did not. The
fields are 256×256 RGB PNGs. Frozen validation `00002` is upstream train row 2;
its current CQ source equals `cadquery_file` byte-for-byte. Expanded 9K
`sample-00000` is upstream test row 0, despite its frozen dataset label
`zero2cad100k-train`; its CQ source also matches byte-for-byte. Four frozen
validation records and two expanded-9K records were source-verified and copied
into bounded `/tmp` image manifests. The producer's field order is
`image_0` through `image_7`. The upstream paper describes four front and four
rear views but does not publish a per-field camera-direction map, so this
integration preserves the published field order exactly.

The fallback canonical STEP/BRep renderer is specified provisionally in
`canonical_render_contract.json`. No fallback fixture was needed because the
original upstream images were accessible. No full 9K image cache or 500K
container was built.

## Verification and remaining gate

The real Qwen2.5-VL processor passed an eight-image fixture. On frozen sample
`00002`, it produced `image_grid_thw=[1,18,18]` per view, 81 merged visual
tokens per view, 648 total, and a 705-token conditioning prefix. Action 2 has
32 command tokens and a 737-token complete action input. The official
`get_rope_index` returned valid `[3,1,737]` MRoPE positions. Structural Stage2
tests passed in the PointerCAD environment, including text-only regression,
future-target isolation, multi-token feedback, zero-pointer and ragged paths.
An official Qwen2.5-VL class with reduced random dimensions also completed a
native V1 prepared-state action (`00002:2`) with eight real images, finite
grammar/pointer/scalar losses and context/feedback gradients. This checks the
integration but is not evidence for the pretrained 3B learnability gate.

The Qwen2.5-VL-3B checkpoint, real forward/backward, gradient check, bounded
micro-overfit and cost probe are recorded in the audit JSON after completion.

## Pretrained 3B gate result

The official Qwen2.5-VL-3B-Instruct checkpoint was downloaded from the publisher mirror into a temporary CCI cache. Both safetensors shards match the official index hashes. The real gate used the H100 HBM3 MIG 3g.40gb slice (42,412,802,048 visible bytes), bf16, frozen base language/vision/merger weights and rank-8 language LoRA.

The smoke used frozen validation record `00002`, native prepared V1 state at action 2 and eight upstream RGB PNG views. The official processor produced 81 merged visual tokens per view and 648 total. The current action contained 32 command tokens and 737 total multimodal tokens. Forward took 0.927 s, backward 0.380 s, peak allocated VRAM was 12.51 GB and peak reserved VRAM was 12.71 GB. The model performed one visual forward per trajectory; the bounded four-trajectory run performed four visual forwards for eight measured actions. The first producing pointer hidden state was unchanged by later feedback (`max delta 0.0`), while a later pointer changed (`max delta 105.25`), and candidate-bank external-key parity held.

All intended groups had trainable parameters; the real smoke produced nonzero gradients in language LoRA, grammar, typed pointer, context, scalar/record, feedback and UVNet/GNN groups. The four-trajectory, eight-epoch micro-overfit reduced total loss from 11.2404 to 1.3694, pointer loss from 0.2732 to 0.0187, grammar loss from 5.5130 to 1.1004, scalar loss from 3.6053 to 0.2308 and record loss from 1.8489 to 0.0196.

The final verdict is `STAGE2_QWEN25VL_READY_FOR_9K_PILOT`. Full 9K image materialization remains a bounded pilot handoff step by design; the task verified exact source mapping and executable eight-view manifests for all four frozen validation records plus two expanded records. Native V2 parity remains an independent handoff gate.
