# Architecture

This repository is organized around one safety rule: residue-level labels are only attached to structure features after sequence-verified mapping.

## Pipeline

1. `import-dyna1-labels`
   - Reads Dyna-1 label artifacts.
   - Uses the entry-level label sequence as the canonical label coordinate system.
   - Aligns labels to manifest entities by sequence.
   - Writes a residue label table and an alignment QC report.

2. `extract-sidechain-features`
   - Reads ESMFold PDB files.
   - Aligns parsed PDB residues back to the manifest sequence.
   - Masks terminal His-tags and unreliable residues.
   - Emits one row per manifest residue.

3. `extract-esm2-embeddings`
   - Pre-extracts frozen `facebook/esm2_t30_150M_UR50D` layer-30 embeddings.
   - Stores one compressed `.npz` file per protein.
   - Saves embeddings as `float16`; training upcasts to `float32`.

4. `build-dataset`
   - Joins feature rows and verified label rows by `protein_id` and `sequence_pos_1based`.

5. `make-splits`
   - Creates protein-level splits from MMseqs2 clusters or a provided cluster table.
   - Excludes clusters similar to held-out evaluation proteins.

6. `train`, `evaluate`, `run-ablations`
   - Provide compact baseline/fusion training and metric reporting.

## Fail-Closed Labeling

The importer never assigns labels by raw residue number or by `bmrb_id` alone. If a label entry cannot be mapped to exactly one manifest entity with sufficient sequence identity and coverage, it is written to QC as `ambiguous` or `unmapped` and excluded from residue-level training/evaluation.

## Optional Dependencies

The core label and feature safety logic runs with `numpy` and `pandas`. Heavy dependencies are optional:

- `mdtraj` for DSSP fallback.
- `torch` and `transformers` for ESM-2 embedding extraction and training.
- `pyarrow` for Parquet output.
