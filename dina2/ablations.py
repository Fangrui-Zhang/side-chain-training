"""Ablation runner for DINA2 model variants."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .training import train_model


def run_ablations(
    dataset_csv: str,
    split_csv: str,
    out_dir: str,
    embedding_index_csv: str | None = None,
    epochs: int = 5,
    seed: int = 13,
) -> pd.DataFrame:
    variants = [
        ("embedding_only", "embedding", 0.0),
        ("sidechain_only", "sidechain", 0.0),
        ("fusion_lambda0", "fusion", 0.0),
        ("fusion_lambda02", "fusion", 0.2),
        ("fusion_lambda1", "fusion", 1.0),
    ]
    rows = []
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, model_type, lambda_missing in variants:
        result = train_model(
            dataset_csv=dataset_csv,
            embedding_index_csv=embedding_index_csv,
            split_csv=split_csv,
            out_dir=root / name,
            model_type=model_type,
            seed=seed,
            epochs=epochs,
            lambda_missing=lambda_missing,
        )
        rows.append({"ablation": name, "model_type": model_type, "lambda_missing": lambda_missing, "out_dir": result["out_dir"]})
    out = pd.DataFrame(rows)
    out.to_csv(root / "ablation_index.csv", index=False)
    return out
