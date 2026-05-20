"""Protein-level split generation with MMseqs2 cluster support."""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def _write_fasta(manifest: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in manifest.itertuples(index=False):
            handle.write(f">{getattr(row, 'protein_id')}\n{getattr(row, 'sequence')}\n")


def _read_cluster_tsv(path: str | Path) -> pd.DataFrame:
    clusters = pd.read_csv(path, sep="\t", header=None, names=["cluster", "protein_id"])
    return clusters


def create_sequence_splits(
    manifest_csv: str | Path,
    out_csv: str | Path,
    cluster_tsv: str | Path | None = None,
    eval_ids_csv: str | Path | None = None,
    min_seq_id: float = 0.30,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 13,
    work_dir: str | Path = "data/mmseqs",
) -> pd.DataFrame:
    """Create protein-level splits from MMseqs2 clusters."""

    manifest = pd.read_csv(manifest_csv)
    required = {"protein_id", "sequence"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    if cluster_tsv is None:
        if shutil.which("mmseqs") is None:
            raise RuntimeError("MMseqs2 is required for split generation. Install mmseqs or pass --cluster-tsv.")
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        fasta = work / "sequences.fasta"
        _write_fasta(manifest, fasta)
        subprocess.run(
            [
                "mmseqs",
                "easy-cluster",
                str(fasta),
                str(work / "clusters"),
                str(work / "tmp"),
                "--cluster-mode",
                "1",
                "--min-seq-id",
                str(min_seq_id),
                "-k",
                "5",
            ],
            check=True,
        )
        cluster_tsv = work / "clusters_cluster.tsv"

    clusters = _read_cluster_tsv(cluster_tsv)
    known = set(manifest["protein_id"].astype(str))
    clusters = clusters[clusters["protein_id"].astype(str).isin(known)]
    if clusters.empty:
        raise ValueError("No manifest proteins found in cluster table")

    eval_ids: set[str] = set()
    if eval_ids_csv is not None:
        eval_df = pd.read_csv(eval_ids_csv)
        if "protein_id" not in eval_df.columns:
            raise ValueError("eval_ids_csv must include protein_id")
        eval_ids = set(eval_df["protein_id"].astype(str))
    contaminated_clusters = set(clusters.loc[clusters["protein_id"].astype(str).isin(eval_ids), "cluster"].astype(str))

    unique_clusters = sorted(set(clusters["cluster"].astype(str)) - contaminated_clusters)
    rng = random.Random(seed)
    rng.shuffle(unique_clusters)
    n_test = int(round(len(unique_clusters) * test_fraction))
    n_val = int(round(len(unique_clusters) * val_fraction))
    test_clusters = set(unique_clusters[:n_test])
    val_clusters = set(unique_clusters[n_test : n_test + n_val])

    rows = []
    for row in clusters.itertuples(index=False):
        cluster = str(getattr(row, "cluster"))
        protein_id = str(getattr(row, "protein_id"))
        if cluster in contaminated_clusters:
            split = "excluded_eval_similar"
        elif cluster in test_clusters:
            split = "test"
        elif cluster in val_clusters:
            split = "val"
        else:
            split = "train"
        rows.append({"protein_id": protein_id, "cluster": cluster, "split": split})
    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out
