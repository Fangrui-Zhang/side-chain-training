import zipfile

import pandas as pd

from dina2.labels import import_cpmg_labels, match_label_to_entity, read_cpmg_labels


def _write_cpmg_zip(path, frame):
    json_path = path.with_suffix(".json")
    json_path.write_text(frame.to_json(), encoding="utf-8")
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(json_path, arcname="RelaxDB_CPMG_22jan2025.json")


def test_read_cpmg_schema(tmp_path):
    zpath = tmp_path / "cpmg.zip"
    frame = pd.DataFrame(
        {
            "entry_ID": ["X_CPMG"],
            "sequence": ["ACDE"],
            "label": ["AXY."],
            "seq len": [4],
            "same len as seq?": [1.0],
        }
    )
    _write_cpmg_zip(zpath, frame)
    out = read_cpmg_labels(zpath)
    assert out.loc[0, "entry_ID"] == "X_CPMG"


def test_import_cpmg_maps_by_sequence_and_masks_histag(tmp_path):
    zpath = tmp_path / "cpmg.zip"
    frame = pd.DataFrame(
        {
            "entry_ID": ["X_CPMG"],
            "sequence": ["ACDE"],
            "label": ["AXY."],
            "seq len": [4],
            "same len as seq?": [1.0],
        }
    )
    _write_cpmg_zip(zpath, frame)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "protein_id": ["bmr1_entity2"],
            "bmrb_id": [1],
            "entity_id": [2],
            "sequence": ["ACDEHHHHHH"],
        }
    ).to_csv(manifest, index=False)
    labels, qc = import_cpmg_labels(zpath, manifest, tmp_path / "labels.csv", tmp_path / "qc.csv")
    assert qc.loc[0, "status"] == "mapped"
    assert labels["protein_id"].unique().tolist() == ["bmr1_entity2"]
    assert labels["sequence_pos_1based"].tolist() == [1, 2, 3, 4]
    assert labels["eval_mask"].tolist() == [1, 1, 1, 1]


def test_ambiguous_entity_is_not_mapped():
    manifest = pd.DataFrame(
        {
            "protein_id": ["bmr1_entity1", "bmr1_entity2"],
            "entity_id": [1, 2],
            "sequence": ["ACDE", "ACDE"],
        }
    )
    match = match_label_to_entity("X", "ACDE", manifest)
    assert match.status == "ambiguous"
    assert match.protein_id is None
