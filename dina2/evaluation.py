"""Evaluation metrics for residue-level predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Evaluation requires scikit-learn. Install with: python -m pip install -e '.[ml]'") from exc

    labels = labels.astype(int)
    if len(np.unique(labels)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))
    baseline = float(labels.mean()) if len(labels) else float("nan")
    return {"AUROC": auroc, "AUPRC": auprc, "AUPRC_norm": auprc - baseline, "baseline": baseline}


def evaluate_predictions(predictions_csv: str, out_csv: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)
    required = {"protein_id", "label", "score", "eval_mask"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction table missing columns: {sorted(missing)}")
    rows = []
    masked = df[df["eval_mask"] == 1]
    overall = binary_metrics(masked["label"].to_numpy(), masked["score"].to_numpy())
    rows.append({"group": "overall", "protein_id": "", **overall})
    for protein_id, group in masked.groupby("protein_id"):
        rows.append({"group": "per_protein", "protein_id": protein_id, **binary_metrics(group["label"].to_numpy(), group["score"].to_numpy())})
    out = pd.DataFrame(rows)
    if out_csv:
        out.to_csv(out_csv, index=False)
    return out
