#!/bin/bash
set -euo pipefail

SHARD_INDEX="${1:?usage: run_extract_esm2.sh SHARD_INDEX NUM_SHARDS}"
NUM_SHARDS="${2:?usage: run_extract_esm2.sh SHARD_INDEX NUM_SHARDS}"

MANIFEST="${DINA2_MANIFEST:-data/manifest/esmfold_pdb_manifest.csv}"
OUT_BASE="${DINA2_OUT_BASE:-/staging/${USER}/dina2-sidechain}"
OUT_DIR="${OUT_BASE}/embeddings/esm2_t30_150M_UR50D_layer30"

mkdir -p "${OUT_DIR}"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Shard ${SHARD_INDEX} of ${NUM_SHARDS}"
echo "Manifest: ${MANIFEST}"
echo "Output: ${OUT_DIR}"

python -m dina2.cli extract-esm2-embeddings \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}" \
  --resume
