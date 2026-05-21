"""ESM-2 embedding pre-extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ESM2_MODEL = "facebook/esm2_t30_150M_UR50D"
ESM2_LAYER = 30
ESM2_DIM = 640


def extract_esm2_embeddings(
    manifest_csv: str | Path,
    out_dir: str | Path,
    model_name: str = ESM2_MODEL,
    layer: int = ESM2_LAYER,
    limit: int | None = None,
    start: int | None = None,
    count: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    """Pre-extract ESM-2 embeddings to one float16 NPZ per protein."""

    manifest = pd.read_csv(manifest_csv)
    manifest = _select_manifest_rows(
        manifest,
        limit=limit,
        start=start,
        count=count,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    index_rows = []
    pending_rows = []
    for manifest_pos, row in enumerate(manifest.itertuples(index=False)):
        protein_id = str(getattr(row, "protein_id"))
        sequence = str(getattr(row, "sequence")).replace(" ", "").upper()
        file_path = out_path / f"{protein_id}.npz"
        index_row = {
            "_manifest_pos": manifest_pos,
            "protein_id": protein_id,
            "sequence_length": len(sequence),
            "embedding_path": str(file_path),
            "model_name": model_name,
            "layer": layer,
            "dtype": "float16",
        }
        if resume and file_path.exists():
            index_rows.append(index_row)
        else:
            pending_rows.append((row, protein_id, sequence, file_path, index_row))

    if not pending_rows:
        index = pd.DataFrame(index_rows).drop(columns=["_manifest_pos"], errors="ignore")
        index.to_csv(out_path / _index_filename(shard_index, num_shards), index=False)
        return index

    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ESM-2 extraction requires optional ML dependencies. "
            "Install with: python -m pip install -e '.[ml]'"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    for _row, protein_id, sequence, file_path, index_row in pending_rows:
        encoded = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        hidden = outputs.hidden_states[layer][0, 1 : 1 + len(sequence), :].detach().cpu().numpy()
        if hidden.shape[0] != len(sequence):
            raise ValueError(f"Embedding length mismatch for {protein_id}: {hidden.shape[0]} != {len(sequence)}")
        file_path = out_path / f"{protein_id}.npz"
        np.savez_compressed(
            file_path,
            embedding=hidden.astype(np.float16),
            sequence=sequence,
            positions_1based=np.arange(1, len(sequence) + 1, dtype=np.int32),
        )
        index_rows.append(index_row)
    index = pd.DataFrame(index_rows).sort_values("_manifest_pos").drop(columns=["_manifest_pos"], errors="ignore")
    index.to_csv(out_path / _index_filename(shard_index, num_shards), index=False)
    return index


def _select_manifest_rows(
    manifest: pd.DataFrame,
    *,
    limit: int | None = None,
    start: int | None = None,
    count: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> pd.DataFrame:
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        manifest = manifest.head(limit)
    if start is not None:
        if start < 0:
            raise ValueError("start must be non-negative")
        manifest = manifest.iloc[start:]
    if count is not None:
        if count < 0:
            raise ValueError("count must be non-negative")
        manifest = manifest.head(count)
    if (shard_index is None) != (num_shards is None):
        raise ValueError("shard-index and num-shards must be provided together")
    if shard_index is not None and num_shards is not None:
        if num_shards <= 0:
            raise ValueError("num-shards must be positive")
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError("shard-index must be in [0, num-shards)")
        manifest = manifest.iloc[shard_index::num_shards]
    return manifest.reset_index(drop=True)


def _index_filename(shard_index: int | None, num_shards: int | None) -> str:
    if shard_index is None or num_shards is None:
        return "embedding_index.csv"
    return f"embedding_index_shard_{shard_index:04d}_of_{num_shards:04d}.csv"
