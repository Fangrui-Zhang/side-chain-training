# Commands

All commands can be run through the module form without installation:

```bash
python -m dina2.cli --help
```

After editable install:

```bash
python -m pip install -e ".[dev]"
dina2 --help
```

## Import CPMG Labels

```bash
dina2 import-dyna1-labels \
  --label-zip data/raw/RelaxDB_CPMG_22jan2025.json.zip \
  --manifest "/Users/fangruizhang/Desktop/NMR and research/esmfold_structures_package/esmfold_pdb_manifest.csv" \
  --out-labels data/labels/cpmg_labels.csv \
  --out-qc data/qc/cpmg_alignment_qc.csv
```

## Extract Sidechain Features

```bash
dina2 extract-sidechain-features \
  --manifest "/Users/fangruizhang/Desktop/NMR and research/esmfold_structures_package/esmfold_pdb_manifest.csv" \
  --pdb-root "/Users/fangruizhang/Desktop/NMR and research/esmfold_structures_package/esmfold_structures" \
  --out data/features/sidechain_features.csv \
  --out-qc data/qc/sidechain_feature_qc.csv
```

Use `--limit 20` for a smoke test.

## Extract ESM-2 Embeddings

```bash
dina2 extract-esm2-embeddings \
  --manifest "/Users/fangruizhang/Desktop/NMR and research/esmfold_structures_package/esmfold_pdb_manifest.csv" \
  --out-dir data/embeddings/esm2_t30_150M_UR50D_layer30
```

This requires:

```bash
python -m pip install -e ".[ml]"
```

## Build Dataset

```bash
dina2 build-dataset \
  --features data/features/sidechain_features.csv \
  --labels data/labels/cpmg_labels.csv \
  --out data/datasets/cpmg_features.csv
```

## Create Splits

With a precomputed MMseqs2 cluster table:

```bash
dina2 make-splits \
  --manifest data/manifest.csv \
  --cluster-tsv data/mmseqs/clusters_cluster.tsv \
  --out data/splits/protein_splits.csv
```

Without a cluster table, install MMseqs2 and omit `--cluster-tsv`.

## Train

```bash
dina2 train \
  --dataset data/datasets/cpmg_features.csv \
  --split data/splits/protein_splits.csv \
  --embedding-index data/embeddings/esm2_t30_150M_UR50D_layer30/embedding_index.csv \
  --model-type fusion \
  --out-dir runs/fusion_smoke \
  --epochs 5
```

## Run Ablations

```bash
dina2 run-ablations \
  --dataset data/datasets/cpmg_features.csv \
  --split data/splits/protein_splits.csv \
  --embedding-index data/embeddings/esm2_t30_150M_UR50D_layer30/embedding_index.csv \
  --out-dir runs/ablations \
  --epochs 5
```
