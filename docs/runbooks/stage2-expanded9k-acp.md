# Stage2 Expanded-9K ACP Runbook

This run is manually launched on ACP. CCI verification must remain bounded.
The launchers call `/root/miniconda/bin/conda` directly, so batch shells do not
need to source `conda.sh`. Set `CONDA_EXE=/absolute/path/to/conda` if the image
uses a different installation path.

## CPU corpus job

```bash
cd /mnt/afs/L202500475/cadquery2crs
CONDA_ENV=cadquery2crs WORKERS=32 \
  scripts/acp/build_stage2_expanded9k_cpu.sh
```

The job writes `/mnt/afs/L202500475/cadquery2crs-corpus/stage2-expanded9k-v1`.
Resume with `RESUME=1` after a recoverable interruption. Continue only when
`release_check.json` reports `EXPANDED_9K_CORPUS_READY`.

## GPU diagnostic job

```bash
cd /mnt/afs/L202500475/Pointer-CAD
CONDA_ENV=PointerCAD NPROC=4 \
  scripts/acp/train_stage2_expanded9k_4xh100.sh
```

The default is one example per device, eight gradient accumulation steps, three
epochs, checkpoints every 250 optimizer steps, and validation/checkpointing at
epoch boundaries. Resume with:

```bash
scripts/acp/train_stage2_expanded9k_4xh100.sh \
  --resume /mnt/afs/L202500475/experiments/stage2-expanded9k-v1/checkpoints/step-XXXXXXXX.pt
```

The launcher performs a 20-step four-rank preflight and exits before the main
run on any NCCL, finite-loss, data, or checkpoint failure. Inspect
`logs/launcher.log`, `metrics.jsonl`, `preflight_results.json`, and `summary.md`.
