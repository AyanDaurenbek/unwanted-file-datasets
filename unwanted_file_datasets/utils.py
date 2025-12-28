from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml


def clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / counts.sum()
    non_zero = probs[probs > 0]
    return float(-np.sum(non_zero * np.log2(non_zero)))


def bounded_normal(mean: float, std: float, low: float, high: float, size: int, rng: np.random.Generator) -> np.ndarray:
    values = rng.normal(loc=mean, scale=std, size=size)
    return np.clip(values, low, high)


def weighted_choice(options: Iterable[str], probabilities: Iterable[float], size: int, rng: np.random.Generator) -> list:
    return rng.choice(list(options), size=size, p=list(probabilities)).tolist()


def save_dataframe(df: pd.DataFrame, output_dir: Path, formats: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        if fmt == "csv":
            df.to_csv(output_dir / "dataset.csv", index=False)
        elif fmt == "json":
            df.to_json(output_dir / "dataset.json", orient="records", lines=True)
        elif fmt == "parquet":
            try:
                df.to_parquet(output_dir / "dataset.parquet", index=False)
            except ImportError as exc:
                raise RuntimeError("Parquet support requires optional dependencies (pyarrow/fastparquet)") from exc
        else:
            raise ValueError(f"Unsupported export format: {fmt}")


def save_yaml(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def describe_numeric(df: pd.DataFrame, numeric_cols: Iterable[str]) -> Dict[str, Dict[str, float]]:
    description: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        if col not in df:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        description[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
        }
    return description


def stratified_split(df: pd.DataFrame, split_ratios: Tuple[float, float, float], seed: int) -> pd.DataFrame:
    train_ratio, val_ratio, test_ratio = split_ratios
    rng = np.random.default_rng(seed)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_assignments = []
    for label, group in df.groupby("class_label"):
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        indices = group.index.tolist()
        rng.shuffle(indices)
        for idx in indices[:train_end]:
            split_assignments.append((idx, "train"))
        for idx in indices[train_end:val_end]:
            split_assignments.append((idx, "val"))
        for idx in indices[val_end:]:
            split_assignments.append((idx, "test"))
    split_map = dict(split_assignments)
    df["split"] = df.index.map(split_map.get)
    return df
