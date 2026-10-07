# Expanded-9K VLM pilot handoff — 2026-10-07

Branch: `stage2-expanded9k-vlm-pilot-handoff`, based on
`9aee5d6bb1ddf21033c194777998e18ad5e9ea41`. This work uses a local
`/tmp` checkout. Frozen CRS membership, Pipeline V2, and ongoing Native V1/V2
release work were read only.

## Image mapping and storage

`tools/certify_stage2_arrow_images.py` checked all 9087 frozen identities in
the expanded corpus manifest. Each `zero2cad100k-train/sample-NNNNN/000/v000`
identity resolves to upstream Zero-to-CAD-100k `test` row `NNNNN`. The proof
compares the original CQ source file byte for byte with Arrow
`cadquery_file`, checks the frozen source SHA as provenance, requires
`num_renders == 8`, and fully decodes `image_0` through `image_7` as
256×256 RGB PNG. Result: **9087 mapped, 0 failed**. The upstream dataset label
in the frozen corpus is retained as part of identity; the upstream `test`
split is only an image locator.

The committed `config/stage2_expanded9k_zero2cad_arrow_index.json` is 3.6 MB
and contains only identities, split/row locators, ordered field names,
dimensions, and CQ provenance digests. No image blobs or 72,696 loose PNGs
were written. The runtime Arrow root is provided separately.
Canonical rendering for future sources, 4M-image storage, sharding, and
CADEvolve/FutureCAD/SynCAD adapters remain post-9K work.

To regenerate the index and certification audit, run:

```bash
CORPUS=/mnt/afs/L202500475/cadquery2crs-corpus/stage2-expanded9k-v1
ZERO2CAD_SOURCE_ROOT=/mnt/afs/L202500475/zero2cad_rp/outputs/test-cadquery-source-operation-sets
ARROW_DATASET_ROOT=/mnt/afs/L202500475/hf-data/datasets/Zero-To-CAD-100k
python tools/certify_stage2_arrow_images.py \
  --frozen "$CORPUS" \
  --source-manifest "$CORPUS/source_manifest.jsonl" \
  --source-root "$ZERO2CAD_SOURCE_ROOT" \
  --dataset-root "$ARROW_DATASET_ROOT" \
  --index config/stage2_expanded9k_zero2cad_arrow_index.json \
  --audit-dir docs/audits/2026-10-07-stage2-expanded9k-vlm-pilot-handoff
```

`ZERO2CAD_SOURCE_ROOT` is the original materialized test source root with
`sources/sample-NNNNN.py`; it is not part of semantic image identity.

## Runtime provider and model path

`Zero2CADArrowImageProvider` opens the upstream `test` split on first use,
selects only CQ provenance and eight image fields, and returns eight PIL
images in field order. The HF dataset is memory mapped per process and reopened
after a DataLoader worker fork/spawn. Every read checks the certified CQ
digest, `num_renders`, format, mode, and dimensions. Missing or corrupt views
raise an error. The existing file backed manifest remains available for debug.

The collator accepts either eight file paths or eight PIL images. Both use the
official Qwen2.5-VL processor content list and preserve identical command
token spans. The VLM visual tower, merger, MRoPE, causal Stage2 state, pointer
feedback, and all losses remain the established production integration.

`scripts/train_stage2_expanded9k.py` accepts `--arrow-image-index` and
`--arrow-dataset-root`. It reads a trajectory's views once and retains the
existing one visual forward per trajectory. For long 9K trajectories, it now
backpropagates each action as it is scored, retains only scalar metrics, and
synchronizes DDP gradients at accumulation boundaries. It shards frozen
training identities by rank, uses the actual per rank optimizer step count
for the LR schedule, bounds periodic validation by record count, and writes
adapter/Stage2 trainable checkpoints rather than duplicating the frozen 3B
weights in every checkpoint.

## Real V1 plumbing smoke

Two individually certified complete V1 records (`sample-00000` and
`sample-00001`) were copied into a temporary local prepared root. The
original V1 output was not changed. Each record passed the frozen identity →
`PreparedStage2Corpus.state_for` → Arrow eight views → official processor →
pretrained Qwen2.5-VL-3B → Stage2 loss/backward path. Both losses were finite,
both pointer bearing actions backpropagated, and the vision tower ran once per
trajectory. Peak reserved memory was 12.75 GB. This is a plumbing smoke; the
earlier four trajectory multimodal micro overfit remains the learnability gate.

The bounded action batch of two was skipped: the current trainer processes
trajectories and actions sequentially, so a real concurrent action batch
would require changes outside this handoff. The earlier representative H100
MIG action reference is 1.3065 seconds forward plus backward, with 1.1559
seconds of visual encoding per trajectory. For 202,839 frozen actions and
9087 trajectories, ideal scaling projects about 76.5 hours for one H100,
19.1 hours per epoch on four H100s, and 57.4 hours for three epochs. Data
loading, DDP communication, validation, checkpointing, and action length
variation remain for ACP measurement.

## ACP launch gate

`scripts/acp/train_stage2_expanded9k_vlm_4xh100.sh` defaults to four GPUs
and supports environment overrides for all runtime roots, model, output,
batching, and accumulation. It reads only frozen CRS, final Native V2 native
manifest/release check, certified Arrow index, and the cached Qwen model. It
fails before creating output if the Native V2 manifest or release check is
absent. The Python preflight checks 9087 unique frozen/native/image identities,
matching frozen SHA, the 9087/9087 certification digest, Arrow split, official
processor/config, and local safetensors shards. It does not rebuild CRS,
Pipeline V2, Native V1, or image files.

The final Native V2 manifest/release check is still being prepared in a
separate effort. The expected next action is to point `PREPARED` at that
released native root and run the ACP launcher manually. The machine readable
readiness file marks Native V2 as the sole remaining pilot gate.
