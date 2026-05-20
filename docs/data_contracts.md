# Data Contracts

## Manifest

The manifest must include:

- `protein_id`: stable ID, preferably `bmr<ID>_entity<N>`.
- `sequence`: one-letter amino-acid sequence.

Recommended columns:

- `bmrb_id`
- `entity_id`
- `pdb_filename`, `pdb_path`, or `pdb_path_relative`

## Dyna-1 CPMG Labels

`RelaxDB_CPMG_22jan2025.json.zip` is expected to contain one JSON file readable by `pandas.read_json`.

Required columns:

- `entry_ID`
- `sequence`
- `label`
- `seq len`
- `same len as seq?`

The `label` string is position-aligned to the entry-level `sequence`. The parser does not reconstruct CPMG labels from residue numbers.

Default CPMG token handling:

- `A`: negative, evaluated.
- `X`: exchange positive, evaluated.
- `Y`: unsuppressed R2 exchange-like signal, negative by default.
- `.`: missing peak positive, evaluated.
- `N`, `P`, `p`, and other tokens are not evaluated unless explicitly handled later.

Use `--include-unsuppressed` to treat `Y` as positive.

## Alignment QC

Every imported label entry emits a QC row with:

- `source_entry_id`
- `protein_id`
- `entity_id`
- `status`: `mapped`, `ambiguous`, or `unmapped`
- `reason`
- `label_coverage`
- `identity`
- `matched_label_positions`
- `reliable_label_positions`
- `mismatches`
- `histag_spans`
- `candidates_passing`

Default acceptance thresholds:

- `min_label_coverage = 0.90`
- `min_identity = 0.90`

## Terminal His-Tags

Terminal His-tags are detected as `H{6,}` at either sequence terminus.

His-tag residues:

- remain in the manifest coordinate system
- are assigned `eval_mask = 0`
- are excluded from reliable alignment coverage

## Feature Table

Feature extraction emits one row per manifest residue with:

- `protein_id`
- `entity_id`
- `sequence_pos_1based`
- `amino_acid`
- `feature_valid_mask`
- sidechain geometry/contact/packing/chi features
- pLDDT from the ESMFold PDB B-factor column
- DSSP code and `dssp_mask`
- SASA columns with `sasa_mask = 0` until a SASA backend is added

GLY and ALA are handled explicitly:

- GLY has no sidechain centroid, spread, or chi angles.
- ALA has a CB-based centroid/spread but no chi angles.
