"""Sidechain feature extraction."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .alignment import evaluate_label_to_target_alignment
from .chemistry import (
    AROMATIC,
    CHARGED,
    CHEMICAL_CLASS,
    CHI_ATOMS,
    HYDROPHOBIC,
    MAX_ASA_TIEN,
    NEGATIVE,
    POLAR,
    POSITIVE,
)
from .histags import detect_terminal_histags
from .pdb import Atom, Residue, parse_pdb, residues_to_sequence


def _coord(atom: Atom) -> np.ndarray:
    return np.array([atom.x, atom.y, atom.z], dtype=float)


def _distance(a: Atom, b: Atom) -> float:
    return float(np.linalg.norm(_coord(a) - _coord(b)))


def _dihedral(a: Atom, b: Atom, c: Atom, d: Atom) -> float:
    p0, p1, p2, p3 = (_coord(x) for x in (a, b, c, d))
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(math.atan2(y, x))


def _centroid(atoms: list[Atom]) -> np.ndarray | None:
    if not atoms:
        return None
    return np.vstack([_coord(atom) for atom in atoms]).mean(axis=0)


def _spread(atoms: list[Atom], center: np.ndarray | None) -> float | None:
    if not atoms or center is None:
        return None
    return float(np.sqrt(np.mean([np.sum((_coord(atom) - center) ** 2) for atom in atoms])))


def _min_residue_distance(a: Residue, b: Residue) -> float:
    best = float("inf")
    atoms_a = [atom for atom in a.atoms.values() if atom.element != "H"]
    atoms_b = [atom for atom in b.atoms.values() if atom.element != "H"]
    for aa in atoms_a:
        for bb in atoms_b:
            best = min(best, _distance(aa, bb))
    return best


def _interaction_counts(residue: Residue, neighbors: list[Residue]) -> dict[str, int]:
    aa = residue.aa
    counts = {
        "hydrophobic_contact_count": 0,
        "polar_contact_count": 0,
        "charged_contact_count": 0,
        "aromatic_contact_count": 0,
        "salt_bridge_count": 0,
        "cation_pi_count": 0,
    }
    for nbr in neighbors:
        naa = nbr.aa
        if naa in HYDROPHOBIC:
            counts["hydrophobic_contact_count"] += 1
        if naa in POLAR:
            counts["polar_contact_count"] += 1
        if naa in CHARGED:
            counts["charged_contact_count"] += 1
        if naa in AROMATIC:
            counts["aromatic_contact_count"] += 1
        if (aa in POSITIVE and naa in NEGATIVE) or (aa in NEGATIVE and naa in POSITIVE):
            counts["salt_bridge_count"] += 1
        if (aa in POSITIVE and naa in AROMATIC) or (aa in AROMATIC and naa in POSITIVE):
            counts["cation_pi_count"] += 1
    return counts


def compute_dssp(pdb_path: str, expected_length: int) -> tuple[list[str], list[int], str]:
    """Compute simplified DSSP, returning codes, mask, and failure reason."""

    if shutil.which("mkdssp"):
        # Parsing mkdssp output robustly is intentionally not hand-rolled yet; mdtraj is safer.
        pass
    try:
        import mdtraj as md  # type: ignore

        traj = md.load_pdb(pdb_path)
        codes = md.compute_dssp(traj, simplified=True).tolist()[0]
        codes = [str(code).replace(" ", "C") for code in codes]
        if len(codes) != expected_length:
            return [""] * expected_length, [0] * expected_length, "dssp_length_mismatch"
        return codes, [1] * expected_length, "ok"
    except Exception as exc:  # pragma: no cover - depends on optional mdtraj
        return [""] * expected_length, [0] * expected_length, f"dssp_failed:{type(exc).__name__}"


def compute_sasa(
    pdb_path: str,
    residues: list[Residue],
) -> tuple[dict[str, dict[str, float | str]], str]:
    """Compute residue SASA with Biopython Shrake-Rupley when available."""

    try:
        from Bio.PDB import PDBParser, ShrakeRupley  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return {}, f"sasa_failed:{type(exc).__name__}"

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", str(pdb_path))
        sr = ShrakeRupley()
        sr.compute(structure, level="A")
        sasa_by_key: dict[tuple[str, int, str], dict[str, float]] = {}
        for chain in structure.get_chains():
            for residue in chain:
                hetflag, resseq, icode = residue.id
                if hetflag.strip():
                    continue
                key = (str(chain.id).strip() or "A", int(resseq), str(icode).strip())
                total = 0.0
                sidechain = 0.0
                for atom in residue:
                    element = (getattr(atom, "element", "") or atom.get_name()[0]).upper()
                    if element == "H":
                        continue
                    atom_sasa = float(getattr(atom, "sasa", 0.0))
                    total += atom_sasa
                    if atom.get_name().strip() not in {"N", "CA", "C", "O", "OXT"}:
                        sidechain += atom_sasa
                sasa_by_key[key] = {"total_sasa": total, "sidechain_sasa": sidechain}

        out: dict[str, dict[str, float | str]] = {}
        for residue in residues:
            key = (residue.chain_id, residue.resseq, residue.icode.strip())
            values = sasa_by_key.get(key)
            if values is None:
                continue
            max_asa = MAX_ASA_TIEN.get(residue.aa)
            rel = values["total_sasa"] / max_asa if max_asa else np.nan
            out[residue.residue_id] = {
                "total_sasa": float(values["total_sasa"]),
                "sidechain_sasa": float(values["sidechain_sasa"]),
                "relative_sasa": float(rel) if np.isfinite(rel) else np.nan,
                "core_surface_flag": "core" if np.isfinite(rel) and rel < 0.20 else "surface",
            }
        return out, "ok"
    except Exception as exc:  # pragma: no cover - depends on optional structure parser
        return {}, f"sasa_failed:{type(exc).__name__}"


def resolve_pdb_path(row: pd.Series, manifest_path: Path, pdb_root: str | None) -> Path:
    if "pdb_path" in row and pd.notna(row["pdb_path"]):
        candidate = Path(str(row["pdb_path"]))
    elif "pdb_filename" in row and pd.notna(row["pdb_filename"]):
        candidate = Path(str(row["pdb_filename"]))
    elif "pdb_path_relative" in row and pd.notna(row["pdb_path_relative"]):
        candidate = Path(str(row["pdb_path_relative"]))
    else:
        candidate = Path(f"{row['protein_id']}.pdb")
    if candidate.is_absolute():
        return candidate
    if pdb_root:
        root_candidate = Path(pdb_root) / candidate.name
        if root_candidate.exists():
            return root_candidate
        return Path(pdb_root) / candidate
    return manifest_path.parent / candidate


def _empty_feature_row(protein_id: str, entity_id: object, sequence_pos: int, aa: str) -> dict[str, object]:
    row = {
        "protein_id": protein_id,
        "entity_id": entity_id,
        "sequence_pos_1based": sequence_pos,
        "amino_acid": aa,
        "feature_valid_mask": 0,
    }
    for col in FEATURE_COLUMNS:
        row.setdefault(col, np.nan)
    return row


FEATURE_COLUMNS = [
    "residue_chemical_class",
    "sidechain_heavy_atom_count",
    "sidechain_centroid_x",
    "sidechain_centroid_y",
    "sidechain_centroid_z",
    "sidechain_centroid_mask",
    "ca_to_sidechain_centroid_distance",
    "sidechain_spread",
    "sidechain_geometry_mask",
    "contact_count_4a",
    "contact_count_6a",
    "contact_count_8a",
    "packing_density_8a",
    "hydrophobic_contact_count",
    "polar_contact_count",
    "charged_contact_count",
    "aromatic_contact_count",
    "salt_bridge_count",
    "cation_pi_count",
    "hydrogen_bond_count",
    "disulfide_flag",
    "plddt",
    "total_sasa",
    "sidechain_sasa",
    "relative_sasa",
    "sasa_mask",
    "core_surface_flag",
    "dssp",
    "dssp_mask",
] + [f"chi{i}_{name}" for i in range(1, 5) for name in ("sin", "cos", "mask")]

AUGMENT_QC_COLUMNS = [
    "protein_id",
    "status",
    "reason",
    "pdb_path",
    "identity",
    "coverage",
    "dssp_reason",
    "sasa_reason",
]

EXTRACT_QC_COLUMNS = [
    "protein_id",
    "pdb_path",
    "manifest_length",
    "pdb_length",
    "alignment_status",
    "alignment_reason",
    "identity",
    "coverage",
    "histag_spans",
    "dssp_reason",
    "sasa_reason",
]


def extract_residue_features(
    protein_id: str,
    entity_id: object,
    manifest_sequence: str,
    pdb_path: str | Path,
    min_label_coverage: float = 0.90,
    min_identity: float = 0.90,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Extract feature rows for one protein, aligned back to manifest sequence."""

    residues = parse_pdb(str(pdb_path))
    pdb_sequence = residues_to_sequence(residues)
    tags = detect_terminal_histags(manifest_sequence)
    _, qc = evaluate_label_to_target_alignment(
        manifest_sequence,
        pdb_sequence,
        min_label_coverage=min_label_coverage,
        min_identity=min_identity,
    )
    qc_row = {
        "protein_id": protein_id,
        "pdb_path": str(pdb_path),
        "manifest_length": len(manifest_sequence),
        "pdb_length": len(pdb_sequence),
        "alignment_status": qc.status,
        "alignment_reason": qc.reason,
        "identity": qc.identity,
        "coverage": qc.label_coverage,
        "histag_spans": tags.to_string(),
    }
    if qc.status != "pass":
        return [
            _empty_feature_row(protein_id, entity_id, i + 1, aa)
            for i, aa in enumerate(manifest_sequence)
        ], qc_row

    alignment, _ = evaluate_label_to_target_alignment(
        manifest_sequence,
        pdb_sequence,
        min_label_coverage=min_label_coverage,
        min_identity=min_identity,
    )
    mapping = alignment.mapping_a_to_b
    dssp_codes, dssp_mask, dssp_reason = compute_dssp(str(pdb_path), len(residues))
    qc_row["dssp_reason"] = dssp_reason
    sasa_by_residue, sasa_reason = compute_sasa(str(pdb_path), residues)
    qc_row["sasa_reason"] = sasa_reason

    distance_cache: dict[tuple[int, int], float] = {}
    for i, res_i in enumerate(residues):
        for j in range(i + 1, len(residues)):
            dist = _min_residue_distance(res_i, residues[j])
            distance_cache[(i, j)] = dist
            distance_cache[(j, i)] = dist

    rows: list[dict[str, object]] = []
    for manifest_i, aa in enumerate(manifest_sequence):
        row = _empty_feature_row(protein_id, entity_id, manifest_i + 1, aa)
        if tags.contains(manifest_i):
            row["feature_valid_mask"] = 0
            rows.append(row)
            continue
        pdb_i = mapping.get(manifest_i)
        if pdb_i is None or pdb_i >= len(residues):
            rows.append(row)
            continue
        residue = residues[pdb_i]
        if residue.aa != aa:
            rows.append(row)
            continue
        row.update(_features_for_residue(residue, residues, pdb_i, distance_cache))
        sasa_values = sasa_by_residue.get(residue.residue_id)
        if sasa_values is not None:
            row.update(sasa_values)
            row["sasa_mask"] = 1
        row["dssp"] = dssp_codes[pdb_i] if dssp_mask[pdb_i] else ""
        row["dssp_mask"] = int(dssp_mask[pdb_i])
        row["feature_valid_mask"] = 1
        rows.append(row)
    return rows, qc_row


def _features_for_residue(
    residue: Residue,
    residues: list[Residue],
    index: int,
    distance_cache: dict[tuple[int, int], float],
) -> dict[str, object]:
    aa = residue.aa
    side_atoms = residue.sidechain_atoms
    center = _centroid(side_atoms)
    ca = residue.atoms.get("CA")
    row: dict[str, object] = {
        "residue_chemical_class": CHEMICAL_CLASS.get(aa, "unknown"),
        "sidechain_heavy_atom_count": len(side_atoms),
        "sidechain_centroid_mask": int(center is not None and aa != "G"),
        "sidechain_geometry_mask": int(center is not None and ca is not None and aa != "G"),
        "hydrogen_bond_count": 0,
        "disulfide_flag": 0,
        "total_sasa": np.nan,
        "sidechain_sasa": np.nan,
        "relative_sasa": np.nan,
        "sasa_mask": 0,
        "core_surface_flag": "",
    }
    if center is not None and aa != "G":
        row["sidechain_centroid_x"] = float(center[0])
        row["sidechain_centroid_y"] = float(center[1])
        row["sidechain_centroid_z"] = float(center[2])
        row["sidechain_spread"] = _spread(side_atoms, center)
        if ca is not None:
            row["ca_to_sidechain_centroid_distance"] = float(np.linalg.norm(_coord(ca) - center))

    plddt_values = [atom.bfactor for atom in residue.atoms.values()]
    row["plddt"] = float(np.mean(plddt_values)) if plddt_values else np.nan

    neighbors_4 = []
    neighbors_6 = []
    neighbors_8 = []
    for j, nbr in enumerate(residues):
        if j == index:
            continue
        dist = distance_cache.get((index, j), float("inf"))
        if dist <= 4.0:
            neighbors_4.append(nbr)
        if dist <= 6.0:
            neighbors_6.append(nbr)
        if dist <= 8.0:
            neighbors_8.append(nbr)
    row["contact_count_4a"] = len(neighbors_4)
    row["contact_count_6a"] = len(neighbors_6)
    row["contact_count_8a"] = len(neighbors_8)
    row["packing_density_8a"] = sum(len(nbr.atoms) for nbr in neighbors_8)
    row.update(_interaction_counts(residue, neighbors_8))

    if aa == "C":
        sg = residue.atoms.get("SG")
        if sg is not None:
            row["disulfide_flag"] = int(
                any(
                    other.aa == "C"
                    and other is not residue
                    and "SG" in other.atoms
                    and _distance(sg, other.atoms["SG"]) <= 2.3
                    for other in residues
                )
            )

    chi_defs = CHI_ATOMS.get(residue.resname.upper(), [])
    for idx in range(1, 5):
        row[f"chi{idx}_sin"] = np.nan
        row[f"chi{idx}_cos"] = np.nan
        row[f"chi{idx}_mask"] = 0
    for idx, atom_names in enumerate(chi_defs[:4], start=1):
        atoms = [residue.atoms.get(name) for name in atom_names]
        if all(atom is not None for atom in atoms):
            angle = _dihedral(*atoms)  # type: ignore[arg-type]
            row[f"chi{idx}_sin"] = float(math.sin(angle))
            row[f"chi{idx}_cos"] = float(math.cos(angle))
            row[f"chi{idx}_mask"] = 1
    return row


def extract_features_from_manifest(
    manifest_csv: str | Path,
    out: str | Path,
    pdb_root: str | None = None,
    limit: int | None = None,
    out_qc: str | Path | None = None,
    checkpoint_every: int = 100,
    resume: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = Path(manifest_csv)
    manifest = pd.read_csv(manifest_path)
    if limit is not None:
        manifest = manifest.head(limit)
    out_path = Path(out)
    qc_path = Path(out_qc) if out_qc is not None else None
    completed: set[str] = set()
    if resume and qc_path is not None and qc_path.exists():
        existing_qc = pd.read_csv(qc_path)
        if "protein_id" in existing_qc.columns:
            completed = set(existing_qc["protein_id"].astype(str))
            manifest = manifest[~manifest["protein_id"].astype(str).isin(completed)]
            print(f"Resuming: skipping {len(completed)} proteins already in {qc_path}")
    elif not resume:
        for path in [out_path, qc_path]:
            if path is not None and path.exists():
                path.unlink()

    rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal rows, qc_rows
        if not rows and not qc_rows:
            return
        if rows:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            feature_chunk = pd.DataFrame(rows).reindex(columns=["protein_id", "entity_id", "sequence_pos_1based", "amino_acid", "feature_valid_mask"] + FEATURE_COLUMNS)
            if str(out_path).endswith(".parquet"):
                # Parquet append semantics vary by engine; keep parquet for small non-streaming runs.
                existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
                pd.concat([existing, feature_chunk], ignore_index=True).to_parquet(out_path, index=False)
            else:
                feature_chunk.to_csv(out_path, index=False, mode="a", header=not out_path.exists())
        if qc_path is not None:
            qc_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(qc_rows).reindex(columns=EXTRACT_QC_COLUMNS).to_csv(
                qc_path,
                index=False,
                mode="a",
                header=not qc_path.exists(),
            )
        rows = []
        qc_rows = []

    total = len(manifest)
    for processed, (_, row) in enumerate(manifest.iterrows(), start=1):
        protein_id = str(row["protein_id"])
        entity_id = row.get("entity_id", protein_id.split("_entity")[-1] if "_entity" in protein_id else "")
        pdb_path = resolve_pdb_path(row, manifest_path, pdb_root)
        if not pdb_path.exists():
            qc_rows.append(
                {
                    "protein_id": protein_id,
                    "pdb_path": str(pdb_path),
                    "alignment_status": "fail",
                    "alignment_reason": "pdb_not_found",
                }
            )
            continue
        feature_rows, qc = extract_residue_features(
            protein_id,
            entity_id,
            str(row["sequence"]).replace(" ", "").upper(),
            pdb_path,
        )
        rows.extend(feature_rows)
        qc_rows.append(qc)
        if checkpoint_every > 0 and processed % checkpoint_every == 0:
            flush()
            print(f"[{processed}/{total}] processed through {protein_id}", flush=True)
    flush()
    features = pd.read_csv(out_path) if out_path.exists() and not str(out_path).endswith(".parquet") else pd.DataFrame()
    qc_df = pd.read_csv(qc_path) if qc_path is not None and qc_path.exists() else pd.DataFrame()
    return features, qc_df


def augment_dssp_sasa_features(
    features_csv: str | Path,
    manifest_csv: str | Path,
    out: str | Path,
    pdb_root: str | None = None,
    out_qc: str | Path | None = None,
    checkpoint_every: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill DSSP/SASA columns in an existing feature table without recomputing contacts."""

    features = pd.read_csv(features_csv)
    manifest_path = Path(manifest_csv)
    manifest = pd.read_csv(manifest_path)
    manifest_by_id = {str(row["protein_id"]): row for _, row in manifest.iterrows()}

    out_path = Path(out)
    qc_path = Path(out_qc) if out_qc is not None else None
    for path in [out_path, qc_path]:
        if path is not None and path.exists():
            path.unlink()

    feature_chunks: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal feature_chunks, qc_rows
        if feature_chunks:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(feature_chunks, ignore_index=True).to_csv(
                out_path,
                index=False,
                mode="a",
                header=not out_path.exists(),
            )
            feature_chunks = []
        if qc_path is not None and qc_rows:
            qc_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(qc_rows).reindex(columns=AUGMENT_QC_COLUMNS).to_csv(
                qc_path,
                index=False,
                mode="a",
                header=not qc_path.exists(),
            )
            qc_rows = []

    groups = list(features.groupby("protein_id", sort=False))
    total = len(groups)
    for processed, (protein_id, group) in enumerate(groups, start=1):
        row = manifest_by_id.get(str(protein_id))
        group = group.copy()
        for text_col in ["dssp", "core_surface_flag"]:
            if text_col in group.columns:
                group[text_col] = group[text_col].astype("object")
        if row is None:
            qc_rows.append({"protein_id": protein_id, "status": "fail", "reason": "missing_manifest_row"})
            feature_chunks.append(group)
            continue
        pdb_path = resolve_pdb_path(row, manifest_path, pdb_root)
        if not pdb_path.exists():
            qc_rows.append({"protein_id": protein_id, "status": "fail", "reason": "pdb_not_found", "pdb_path": str(pdb_path)})
            feature_chunks.append(group)
            continue

        residues = parse_pdb(str(pdb_path))
        pdb_sequence = residues_to_sequence(residues)
        manifest_sequence = str(row["sequence"]).replace(" ", "").upper()
        alignment, align_qc = evaluate_label_to_target_alignment(manifest_sequence, pdb_sequence)
        if align_qc.status != "pass":
            qc_rows.append(
                {
                    "protein_id": protein_id,
                    "status": "fail",
                    "reason": align_qc.reason,
                    "pdb_path": str(pdb_path),
                    "identity": align_qc.identity,
                    "coverage": align_qc.label_coverage,
                }
            )
            feature_chunks.append(group)
            continue

        mapping = alignment.mapping_a_to_b
        dssp_codes, dssp_mask, dssp_reason = compute_dssp(str(pdb_path), len(residues))
        sasa_by_residue, sasa_reason = compute_sasa(str(pdb_path), residues)
        for idx, feature_row in group.iterrows():
            manifest_i = int(feature_row["sequence_pos_1based"]) - 1
            pdb_i = mapping.get(manifest_i)
            if pdb_i is None or pdb_i >= len(residues) or int(feature_row.get("feature_valid_mask", 0)) != 1:
                continue
            residue = residues[pdb_i]
            if dssp_mask[pdb_i]:
                group.loc[idx, "dssp"] = dssp_codes[pdb_i]
                group.loc[idx, "dssp_mask"] = 1
            sasa_values = sasa_by_residue.get(residue.residue_id)
            if sasa_values is not None:
                for key, value in sasa_values.items():
                    group.loc[idx, key] = value
                group.loc[idx, "sasa_mask"] = 1
        qc_rows.append(
            {
                "protein_id": protein_id,
                "status": "ok",
                "reason": "ok",
                "pdb_path": str(pdb_path),
                "dssp_reason": dssp_reason,
                "sasa_reason": sasa_reason,
            }
        )
        feature_chunks.append(group)
        if checkpoint_every > 0 and processed % checkpoint_every == 0:
            flush()
            print(f"[{processed}/{total}] augmented through {protein_id}", flush=True)
    flush()
    out_features = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    out_qc_df = pd.read_csv(qc_path) if qc_path is not None and qc_path.exists() else pd.DataFrame()
    return out_features, out_qc_df
