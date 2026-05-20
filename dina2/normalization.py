"""Train-only normalization persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Normalizer:
    columns: list[str]
    mean: dict[str, float]
    std: dict[str, float]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in self.columns:
            value = pd.to_numeric(out[col], errors="coerce").fillna(self.mean[col])
            out[col] = (value - self.mean[col]) / self.std[col]
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


def fit_normalizer(frame: pd.DataFrame, columns: list[str]) -> Normalizer:
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        col_mean = float(values.mean()) if values.notna().any() else 0.0
        col_std = float(values.std(ddof=0)) if values.notna().any() else 1.0
        if not np.isfinite(col_std) or col_std == 0.0:
            col_std = 1.0
        mean[col] = col_mean
        std[col] = col_std
    return Normalizer(columns=columns, mean=mean, std=std)
