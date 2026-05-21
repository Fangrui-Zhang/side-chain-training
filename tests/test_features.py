import pandas as pd

from dina2.features import augment_dssp_sasa_features, extract_features_from_manifest


PDB_TEXT = """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.50           N
ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.50           C
ATOM      3  C   GLY A   1       1.500   1.000   0.000  1.00  0.50           C
ATOM      4  O   GLY A   1       1.500   2.000   0.000  1.00  0.50           O
ATOM      5  N   ALA A   2       2.500   1.000   0.000  1.00  0.70           N
ATOM      6  CA  ALA A   2       3.000   2.000   0.000  1.00  0.70           C
ATOM      7  C   ALA A   2       4.000   2.000   0.000  1.00  0.70           C
ATOM      8  O   ALA A   2       4.500   3.000   0.000  1.00  0.70           O
ATOM      9  CB  ALA A   2       3.000   2.000   1.500  1.00  0.70           C
END
"""


def test_gly_ala_sidechain_masks(tmp_path):
    pdb = tmp_path / "toy.pdb"
    pdb.write_text(PDB_TEXT, encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "protein_id": ["toy"],
            "entity_id": [1],
            "sequence": ["GA"],
            "pdb_filename": ["toy.pdb"],
        }
    ).to_csv(manifest, index=False)
    features, qc = extract_features_from_manifest(manifest, tmp_path / "features.csv", pdb_root=str(tmp_path), out_qc=tmp_path / "qc.csv")
    assert qc.loc[0, "alignment_status"] == "pass"
    gly = features[features["sequence_pos_1based"] == 1].iloc[0]
    ala = features[features["sequence_pos_1based"] == 2].iloc[0]
    assert gly["sidechain_centroid_mask"] == 0
    assert gly["sidechain_geometry_mask"] == 0
    assert ala["sidechain_centroid_mask"] == 1
    assert ala["sidechain_geometry_mask"] == 1
    assert ala["chi1_mask"] == 0


def test_augment_qc_uses_stable_columns_when_first_group_fails(tmp_path):
    pdb = tmp_path / "toy.pdb"
    pdb.write_text(PDB_TEXT, encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "protein_id": ["toy"],
            "entity_id": [1],
            "sequence": ["GA"],
            "pdb_filename": ["toy.pdb"],
        }
    ).to_csv(manifest, index=False)
    features = tmp_path / "features.csv"
    pd.DataFrame(
        [
            {"protein_id": "missing", "sequence_pos_1based": 1, "feature_valid_mask": 1},
            {"protein_id": "toy", "sequence_pos_1based": 1, "feature_valid_mask": 1},
            {"protein_id": "toy", "sequence_pos_1based": 2, "feature_valid_mask": 1},
        ]
    ).to_csv(features, index=False)

    _, qc = augment_dssp_sasa_features(
        features,
        manifest,
        tmp_path / "augmented.csv",
        pdb_root=str(tmp_path),
        out_qc=tmp_path / "augment_qc.csv",
        checkpoint_every=1,
    )

    reread_qc = pd.read_csv(tmp_path / "augment_qc.csv")
    assert list(reread_qc.columns) == [
        "protein_id",
        "status",
        "reason",
        "pdb_path",
        "identity",
        "coverage",
        "dssp_reason",
        "sasa_reason",
    ]
    assert qc["status"].tolist() == ["fail", "ok"]
