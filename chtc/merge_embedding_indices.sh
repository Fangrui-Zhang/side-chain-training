#!/bin/bash
set -euo pipefail

OUT_DIR="${1:-/staging/${USER}/dina2-sidechain/embeddings/esm2_t30_150M_UR50D_layer30}"
OUT_INDEX="${OUT_DIR}/embedding_index.csv"

first=1
tmp="${OUT_INDEX}.tmp"
: > "${tmp}"

for shard_index in "${OUT_DIR}"/embedding_index_shard_*.csv; do
  if [[ ! -e "${shard_index}" ]]; then
    echo "No shard index files found in ${OUT_DIR}" >&2
    exit 1
  fi
  if [[ "${first}" -eq 1 ]]; then
    cat "${shard_index}" >> "${tmp}"
    first=0
  else
    tail -n +2 "${shard_index}" >> "${tmp}"
  fi
done

mv "${tmp}" "${OUT_INDEX}"
echo "Wrote ${OUT_INDEX}"
