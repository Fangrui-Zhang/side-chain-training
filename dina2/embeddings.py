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
) -> pd.DataFrame:
    """Pre-extract ESM-2 embeddings to one float16 NPZ per protein."""

    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ESM-2 extraction requires optional ML dependencies. "
            "Install with: python -m pip install -e '.[ml]'"
        ) from exc

    manifest = pd.read_csv(manifest_csv)
    if limit is not None:
        manifest = manifest.head(limit)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    index_rows = []
    for row in manifest.itertuples(index=False):
        protein_id = str(getattr(row, "protein_id"))
        sequence = str(getattr(row, "sequence")).replace(" ", "").upper()
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
        index_rows.append(
            {
                "protein_id": protein_id,
                "sequence_length": len(sequence),
                "embedding_path": str(file_path),
                "model_name": model_name,
                "layer": layer,
                "dtype": "float16",
            }
        )
    index = pd.DataFrame(index_rows)
    index.to_csv(out_path / "embedding_index.csv", index=False)
    return index
