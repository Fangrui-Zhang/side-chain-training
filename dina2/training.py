"""Small training implementation for DINA2 baselines and fusion model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .normalization import fit_normalizer
from .repro import seed_everything, write_run_metadata


DEFAULT_NUMERIC_FEATURES = [
    "sidechain_heavy_atom_count",
    "ca_to_sidechain_centroid_distance",
    "sidechain_spread",
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
    "disulfide_flag",
    "plddt",
    "total_sasa",
    "sidechain_sasa",
    "relative_sasa",
    "sasa_mask",
] + [f"chi{i}_{name}" for i in range(1, 5) for name in ("sin", "cos", "mask")]


def _require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore

        return torch, nn, Dataset, DataLoader
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Training requires torch. Install with: python -m pip install -e '.[ml]'") from exc


def _load_embeddings(embedding_index: pd.DataFrame) -> dict[str, np.ndarray]:
    embeddings = {}
    for row in embedding_index.itertuples(index=False):
        arr = np.load(getattr(row, "embedding_path"))["embedding"].astype(np.float32)
        embeddings[str(getattr(row, "protein_id"))] = arr
    return embeddings


class ResidueDataset:  # Real base class is injected dynamically to avoid importing torch at module import time.
    pass


def train_model(
    dataset_csv: str | Path,
    embedding_index_csv: str | Path | None,
    split_csv: str | Path,
    out_dir: str | Path,
    model_type: str = "fusion",
    seed: int = 13,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    lambda_missing: float = 0.2,
) -> dict[str, object]:
    """Train a compact MLP baseline/fusion model on residue rows."""

    torch, nn, Dataset, DataLoader = _require_torch()
    seed_everything(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(dataset_csv)
    split = pd.read_csv(split_csv)
    train_ids = set(split.loc[split["split"] == "train", "protein_id"].astype(str))
    train_df = data[data["protein_id"].astype(str).isin(train_ids) & (data["eval_mask"] == 1)].copy()
    if train_df.empty:
        raise ValueError("No training residues with eval_mask=1")

    feature_cols = [col for col in DEFAULT_NUMERIC_FEATURES if col in train_df.columns]
    normalizer = fit_normalizer(train_df, feature_cols)
    normalizer.save(out / "normalizer.json")
    train_df = normalizer.transform(train_df)
    embeddings = _load_embeddings(pd.read_csv(embedding_index_csv)) if embedding_index_csv else {}

    class _ResidueDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, idx: int):
            row = self.frame.iloc[idx]
            feats = row[feature_cols].astype(float).fillna(0.0).to_numpy(dtype=np.float32)
            emb = np.zeros((0,), dtype=np.float32)
            if model_type in {"embedding", "fusion"}:
                protein_id = str(row["protein_id"])
                pos = int(row["sequence_pos_1based"]) - 1
                emb = embeddings[protein_id][pos].astype(np.float32)
            if model_type == "embedding":
                x = emb
            elif model_type == "sidechain":
                x = feats
            else:
                x = np.concatenate([emb, feats]).astype(np.float32)
            return torch.from_numpy(x), torch.tensor(float(row["label"]), dtype=torch.float32)

    loader = DataLoader(_ResidueDataset(train_df), batch_size=batch_size, shuffle=True)
    first_x, _ = next(iter(loader))
    model = nn.Sequential(
        nn.Linear(first_x.shape[1], 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    model.train()
    for epoch in range(epochs):
        losses = []
        for x, y in loader:
            logits = model(x).squeeze(-1)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    torch.save({"model_state_dict": model.state_dict(), "model_type": model_type, "feature_cols": feature_cols}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    write_run_metadata(
        out / "run_metadata.json",
        {
            "dataset_csv": str(dataset_csv),
            "embedding_index_csv": str(embedding_index_csv),
            "split_csv": str(split_csv),
            "model_type": model_type,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lambda_missing": lambda_missing,
        },
    )
    return {"history": history, "feature_cols": feature_cols, "out_dir": str(out)}
