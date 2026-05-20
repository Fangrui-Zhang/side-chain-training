"""Dyna-1 label import and strict entity mapping."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from .alignment import evaluate_label_to_target_alignment
from .histags import detect_terminal_histags


CPMG_LABEL_MAP_DEFAULT = {
    "A": 0,
    "X": 1,
    "Y": 0,
    ".": 1,
}

CPMG_EVAL_TOKENS_DEFAULT = {"A", "X", "Y", "."}


@dataclass(frozen=True)
class EntityMatch:
    source_entry_id: str
    protein_id: str | None
    entity_id: str | int | None
    status: str
    reason: str
    label_coverage: float
    identity: float
    matched_label_positions: int
    reliable_label_positions: int
    mismatches: int
    label_length: int
    target_length: int
    histag_spans: str
    candidates_passing: int


def _read_json_zip(path: str | Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".json")]
        if len(names) != 1:
            raise ValueError(f"Expected exactly one JSON file in {path}, found {names}")
        raw = archive.read(names[0]).decode("utf-8")
    return pd.read_json(io.StringIO(raw))


def read_cpmg_labels(path: str | Path) -> pd.DataFrame:
    """Read and validate Dyna-1 RelaxDB CPMG label ZIP."""

    df = _read_json_zip(path)
    required = {"entry_ID", "sequence", "label", "seq len", "same len as seq?"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CPMG label file missing required columns: {sorted(missing)}")
    bad_rows = []
    for row in df.itertuples(index=False):
        sequence = str(getattr(row, "sequence"))
        label = str(getattr(row, "label"))
        entry = str(getattr(row, "entry_ID"))
        if len(sequence) != len(label):
            bad_rows.append((entry, len(sequence), len(label)))
    if bad_rows:
        raise ValueError(f"CPMG sequence/label length mismatch: {bad_rows[:5]}")
    return df


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"protein_id", "sequence"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if "entity_id" not in manifest.columns:
        manifest["entity_id"] = manifest["protein_id"].astype(str).str.extract(r"_entity(\d+)$")[0]
    return manifest


def _sequence_candidate_subset(manifest: pd.DataFrame, label_sequence: str) -> pd.DataFrame:
    label_len = len(label_sequence)
    sequences = manifest["sequence"].astype(str).str.replace(" ", "", regex=False).str.upper()
    cleaned = sequences.map(_strip_terminal_histags)
    exact = manifest[cleaned == _strip_terminal_histags(label_sequence)]
    if len(exact) > 0:
        return exact

    lengths = cleaned.str.len()
    candidates = manifest[(lengths >= max(1, int(label_len * 0.75))) & (lengths <= int(label_len * 1.35) + 25)].copy()
    if len(candidates) <= 250:
        return candidates

    kmers = {label_sequence[i : i + 6] for i in range(max(0, len(label_sequence) - 5))}

    def quick_score(seq: str) -> float:
        seq = _strip_terminal_histags(seq.replace(" ", "").upper())
        shared = any(kmer in seq for kmer in kmers) if kmers else True
        if not shared:
            return 0.0
        return SequenceMatcher(None, label_sequence, seq).quick_ratio()

    candidates["_quick_score"] = candidates["sequence"].astype(str).map(quick_score)
    candidates = candidates[candidates["_quick_score"] > 0.50]
    return candidates.sort_values("_quick_score", ascending=False).head(250).drop(columns=["_quick_score"])


def _strip_terminal_histags(sequence: str) -> str:
    tags = detect_terminal_histags(sequence)
    start = tags.n_terminal[1] if tags.n_terminal else 0
    end = tags.c_terminal[0] if tags.c_terminal else len(sequence)
    return sequence[start:end]


def _qc_metrics(qc: object) -> dict[str, object]:
    payload = asdict(qc)
    payload.pop("status", None)
    payload.pop("reason", None)
    return payload


def match_label_to_entity(
    source_entry_id: str,
    label_sequence: str,
    manifest: pd.DataFrame,
    explicit_entity_id: str | int | None = None,
    min_label_coverage: float = 0.90,
    min_identity: float = 0.90,
) -> EntityMatch:
    """Match a label sequence to exactly one manifest entity."""

    candidates = manifest
    if explicit_entity_id is not None and not pd.isna(explicit_entity_id):
        candidates = candidates[candidates["entity_id"].astype(str) == str(explicit_entity_id)]
    else:
        candidates = _sequence_candidate_subset(manifest, label_sequence)

    passing: list[tuple[pd.Series, object]] = []
    best: tuple[pd.Series | None, object | None] = (None, None)
    best_key = (-1.0, -1.0)
    for _, row in candidates.iterrows():
        _, qc = evaluate_label_to_target_alignment(
            label_sequence,
            str(row["sequence"]),
            min_label_coverage=min_label_coverage,
            min_identity=min_identity,
        )
        key = (qc.label_coverage, qc.identity)
        if key > best_key:
            best = (row, qc)
            best_key = key
        if qc.status == "pass":
            passing.append((row, qc))

    if len(passing) == 1:
        row, qc = passing[0]
        return EntityMatch(
            source_entry_id=source_entry_id,
            protein_id=str(row["protein_id"]),
            entity_id=row.get("entity_id"),
            status="mapped",
            reason="ok",
            candidates_passing=1,
            **_qc_metrics(qc),
        )
    if len(passing) > 1:
        row, qc = passing[0]
        return EntityMatch(
            source_entry_id=source_entry_id,
            protein_id=None,
            entity_id=None,
            status="ambiguous",
            reason="multiple_entities_pass",
            candidates_passing=len(passing),
            **_qc_metrics(qc),
        )

    row, qc = best
    if qc is None:
        return EntityMatch(
            source_entry_id=source_entry_id,
            protein_id=None,
            entity_id=None,
            status="unmapped",
            reason="no_candidate_entities",
            label_coverage=0.0,
            identity=0.0,
            matched_label_positions=0,
            reliable_label_positions=0,
            mismatches=0,
            label_length=len(label_sequence),
            target_length=0,
            histag_spans="",
            candidates_passing=0,
        )
    return EntityMatch(
        source_entry_id=source_entry_id,
        protein_id=None,
        entity_id=None,
        status="unmapped",
        reason=qc.reason,
        candidates_passing=0,
        **_qc_metrics(qc),
    )


def _label_rows_for_match(
    source_entry_id: str,
    sequence: str,
    label_string: str,
    match: EntityMatch,
    manifest_row: pd.Series | None,
    min_label_coverage: float,
    min_identity: float,
    include_unsuppressed: bool = False,
) -> list[dict[str, object]]:
    if match.status != "mapped" or manifest_row is None:
        return []

    target_sequence = str(manifest_row["sequence"])
    alignment, qc = evaluate_label_to_target_alignment(
        sequence,
        target_sequence,
        min_label_coverage=min_label_coverage,
        min_identity=min_identity,
    )
    if qc.status != "pass":
        return []
    mapping = alignment.mapping_a_to_b
    tags = detect_terminal_histags(target_sequence)
    label_map = dict(CPMG_LABEL_MAP_DEFAULT)
    if include_unsuppressed:
        label_map["Y"] = 1
    eval_tokens = set(CPMG_EVAL_TOKENS_DEFAULT)

    rows = []
    for label_i, aa in enumerate(sequence):
        target_i = mapping.get(label_i)
        raw = label_string[label_i]
        reliable = (
            target_i is not None
            and not tags.contains(target_i)
            and target_sequence[target_i].upper() == aa.upper()
            and raw in eval_tokens
        )
        rows.append(
            {
                "label_source": "dyna1_relaxdb_cpmg",
                "source_entry_id": source_entry_id,
                "protein_id": match.protein_id,
                "entity_id": match.entity_id,
                "sequence_pos_1based": target_i + 1 if target_i is not None else None,
                "aa": aa,
                "raw_label_token": raw,
                "label": label_map.get(raw),
                "label_type": "cpmg",
                "eval_mask": int(bool(reliable and raw in label_map)),
            }
        )
    return rows


def import_cpmg_labels(
    label_zip: str | Path,
    manifest_csv: str | Path,
    out_labels: str | Path,
    out_qc: str | Path,
    min_label_coverage: float = 0.90,
    min_identity: float = 0.90,
    include_unsuppressed: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Import Dyna-1 CPMG labels, map entities by sequence, and write outputs."""

    labels_df = read_cpmg_labels(label_zip)
    manifest = load_manifest(manifest_csv)
    label_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    for row in labels_df.itertuples(index=False):
        entry = str(getattr(row, "entry_ID"))
        sequence = str(getattr(row, "sequence")).replace(" ", "").upper()
        label_string = str(getattr(row, "label")).replace(" ", "")
        explicit_entity = getattr(row, "entity_id", None) if hasattr(row, "entity_id") else None
        match = match_label_to_entity(
            entry,
            sequence,
            manifest,
            explicit_entity_id=explicit_entity,
            min_label_coverage=min_label_coverage,
            min_identity=min_identity,
        )
        qc_rows.append(asdict(match))
        manifest_row = None
        if match.status == "mapped":
            subset = manifest[manifest["protein_id"].astype(str) == str(match.protein_id)]
            manifest_row = subset.iloc[0] if len(subset) else None
        label_rows.extend(
            _label_rows_for_match(
                entry,
                sequence,
                label_string,
                match,
                manifest_row,
                min_label_coverage,
                min_identity,
                include_unsuppressed=include_unsuppressed,
            )
        )

    out_labels = Path(out_labels)
    out_qc = Path(out_qc)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    out_qc.parent.mkdir(parents=True, exist_ok=True)
    label_out = pd.DataFrame(label_rows)
    qc_out = pd.DataFrame(qc_rows)
    label_out.to_csv(out_labels, index=False)
    qc_out.to_csv(out_qc, index=False)
    schema = {
        "source": str(label_zip),
        "format": "column-oriented pandas JSON inside ZIP",
        "required_fields": ["entry_ID", "sequence", "label", "seq len", "same len as seq?"],
        "alignment": {
            "min_label_coverage": min_label_coverage,
            "min_identity": min_identity,
            "terminal_histag_regex": "H{6,}",
        },
    }
    out_qc.with_suffix(".schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return label_out, qc_out


def build_joined_dataset(features_csv: str | Path, labels_csv: str | Path, out: str | Path) -> pd.DataFrame:
    """Join feature and label tables by verified protein/position keys."""

    features = pd.read_csv(features_csv)
    labels = pd.read_csv(labels_csv)
    keys = ["protein_id", "sequence_pos_1based"]
    missing_feature_keys = set(keys) - set(features.columns)
    missing_label_keys = set(keys) - set(labels.columns)
    if missing_feature_keys or missing_label_keys:
        raise ValueError(f"Missing join keys: features={missing_feature_keys}, labels={missing_label_keys}")
    joined = features.merge(labels, on=keys, how="left", suffixes=("", "_label"))
    if "eval_mask" not in joined.columns:
        joined["eval_mask"] = 0
    joined["eval_mask"] = joined["eval_mask"].fillna(0).astype(int)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out, index=False)
    return joined
