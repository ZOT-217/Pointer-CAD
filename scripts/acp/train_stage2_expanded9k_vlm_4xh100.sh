#!/usr/bin/env bash
set -euo pipefail

# ACP only: consume the frozen CRS, final Native V2 release, and upstream Arrow.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
AFS_ROOT="${AFS_ROOT:-/mnt/afs/L202500475}"
CORPUS="${CORPUS:-${AFS_ROOT}/cadquery2crs-corpus/stage2-expanded9k-v1}"
PREPARED="${PREPARED:-${CORPUS}/native}"
ARROW_DATASET_ROOT="${ARROW_DATASET_ROOT:-${AFS_ROOT}/hf-data/datasets/Zero-To-CAD-100k}"
IMAGE_INDEX="${IMAGE_INDEX:-${ROOT}/config/stage2_expanded9k_zero2cad_arrow_index.json}"
IMAGE_CERTIFICATION="${IMAGE_CERTIFICATION:-${ROOT}/docs/audits/2026-10-07-stage2-expanded9k-vlm-pilot-handoff/image_mapping_9087.json}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
OUTPUT="${OUTPUT:-${AFS_ROOT}/experiments/stage2-expanded9k-vlm-v2}"
CONDA_ENV="${CONDA_ENV:-PointerCAD}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda/bin/conda}"
NPROC="${NPROC:-4}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${CONDA_EXE}" ]] || { echo "Conda executable not found: ${CONDA_EXE}" >&2; exit 1; }
[[ "${NPROC}" =~ ^[1-9][0-9]*$ ]] || { echo "NPROC must be a positive integer" >&2; exit 1; }
[[ -f "${PREPARED}/manifest.json" ]] || { echo "Final Native V2 manifest missing: ${PREPARED}" >&2; exit 1; }
[[ -f "${CORPUS}/release_check.json" ]] || { echo "Final Native V2 release_check missing: ${CORPUS}" >&2; exit 1; }
"${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python \
  "${ROOT}/scripts/check_stage2_expanded9k_vlm_launch.py" \
  --corpus "${CORPUS}" --native "${PREPARED}" \
  --image-index "${IMAGE_INDEX}" --image-certification "${IMAGE_CERTIFICATION}" \
  --arrow-dataset-root "${ARROW_DATASET_ROOT}" \
  --model "${MODEL}"

mkdir -p "${OUTPUT}/logs"
exec > >(tee -a "${OUTPUT}/logs/launcher.log") 2>&1
cd "${ROOT}"
exec "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" torchrun --standalone --nproc_per_node="${NPROC}" \
  scripts/train_stage2_expanded9k.py --corpus "${CORPUS}" --prepared "${PREPARED}" \
  --output "${OUTPUT}" --model "${MODEL}" --stage2-conditioning multiview_vlm \
  --arrow-image-index "${IMAGE_INDEX}" --arrow-dataset-root "${ARROW_DATASET_ROOT}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" "$@"
