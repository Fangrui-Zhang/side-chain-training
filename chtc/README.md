# CHTC ESM-2 Embedding Extraction

This folder contains the CHTC scaffolding for extracting frozen ESM-2 embeddings on GPU workers.

## Build The Container

On the CHTC login node:

```bash
cd ~/dina2-sidechain/chtc
condor_submit -i apptainer-build.sub
apptainer build dina2_esm2.sif embedding.def
mv dina2_esm2.sif /staging/$USER/dina2_esm2.sif
exit
```

The definition file downloads `facebook/esm2_t30_150M_UR50D` into the image during build and sets Transformers to offline mode for jobs.

## Submit Embedding Jobs

From the project root on CHTC:

```bash
mkdir -p chtc/logs
condor_submit chtc/extract-esm2.sub
```

The submit file launches 20 GPU shards. Each shard writes `.npz` files and one shard index to:

```bash
/staging/$USER/dina2-sidechain/embeddings/esm2_t30_150M_UR50D_layer30
```

After all shards complete:

```bash
bash chtc/merge_embedding_indices.sh
```

This writes the combined `embedding_index.csv` used by `dina2 train`.
