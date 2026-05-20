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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = Path(manifest_csv)
    manifest = pd.read_csv(manifest_path)
    if limit is not None:
        manifest = manifest.head(limit)
    rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
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

    features = pd.DataFrame(rows)
    qc_df = pd.DataFrame(qc_rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if str(out).endswith(".parquet"):
        features.to_parquet(out, index=False)
    else:
        features.to_csv(out, index=False)
    if out_qc is not None:
        Path(out_qc).parent.mkdir(parents=True, exist_ok=True)
        qc_df.to_csv(out_qc, index=False)
    return features, qc_df
