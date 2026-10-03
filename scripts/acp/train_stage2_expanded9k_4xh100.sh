#!/usr/bin/env bash
set -euo pipefail

# ACP-GPU: consume only the frozen CRS/supervision/native sidecars.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
AFS_ROOT="${AFS_ROOT:-/mnt/afs/L202500475}"
CORPUS="${CORPUS:-${AFS_ROOT}/cadquery2crs-corpus/stage2-expanded9k-v1}"
PREPARED="${PREPARED:-${CORPUS}/native}"
OUTPUT="${OUTPUT:-${AFS_ROOT}/experiments/stage2-expanded9k-v1}"
MODEL="${MODEL:-${AFS_ROOT}/hf-data/models/Qwen2.5-0.5B-Instruct}"
CONDA_ENV="${CONDA_ENV:-PointerCAD}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda/bin/conda}"
NPROC="${NPROC:-4}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:+:${PYTHONPATH}}"
[[ -x "${CONDA_EXE}" ]] || { echo "Conda executable not found: ${CONDA_EXE}; set CONDA_EXE to its absolute path" >&2; exit 1; }
[[ -f "${CORPUS}/release_check.json" ]] || { echo "missing CPU release check: ${CORPUS}" >&2; exit 1; }
python - "${CORPUS}/release_check.json" <<'PY'
import json,sys
if json.load(open(sys.argv[1]))['verdict'] != 'EXPANDED_9K_CORPUS_READY': raise SystemExit('frozen corpus is not READY')
PY
mkdir -p "${OUTPUT}/logs"
exec > >(tee -a "${OUTPUT}/logs/launcher.log") 2>&1
cd "${ROOT}"
exec "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" torchrun --standalone --nproc_per_node="${NPROC}" \
  scripts/train_stage2_expanded9k.py --corpus "${CORPUS}" --prepared "${PREPARED}" --output "${OUTPUT}" \
  --model "${MODEL}" "$@"
