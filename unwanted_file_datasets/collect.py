from __future__ import annotations

import hashlib
import mimetypes
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_CLASSES, RealSourceConfig
from .utils import clamp_ratio, compute_entropy


class RealDataCollector:
    """Collects safe features from files on disk."""

    def __init__(self, context_defaults: Optional[Dict[str, str]] = None) -> None:
        self.context_defaults = context_defaults or {
            "source_channel": "filesystem",
            "user_privilege_level": "unknown",
            "access_time_category": "recent",
        }

    def collect_from_config(self, config: RealSourceConfig) -> pd.DataFrame:
        return self.collect(
            paths=config.paths,
            extensions=config.extensions,
            min_size_bytes=config.min_size_bytes,
            max_size_bytes=config.max_size_bytes,
            exclude_paths=config.exclude_paths,
        )

    def collect(
        self,
        paths: Iterable[Path],
        extensions: Optional[List[str]] = None,
        min_size_bytes: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
        exclude_paths: Optional[Iterable[Path]] = None,
    ) -> pd.DataFrame:
        rows = []
        exclude_set = {Path(p).resolve() for p in (exclude_paths or [])}
        for base in paths:
            base_path = Path(base).resolve()
            if base_path in exclude_set:
                continue
            if base_path.is_file():
                maybe_row = self._extract_features(base_path, extensions, min_size_bytes, max_size_bytes)
                if maybe_row:
                    rows.append(maybe_row)
                continue
            for root, _, files in os.walk(base_path):
                for name in files:
                    file_path = Path(root) / name
                    if file_path in exclude_set:
                        continue
                    maybe_row = self._extract_features(file_path, extensions, min_size_bytes, max_size_bytes)
                    if maybe_row:
                        rows.append(maybe_row)
        return pd.DataFrame(rows)

    def _extract_features(
        self,
        file_path: Path,
        extensions: Optional[List[str]],
        min_size_bytes: Optional[int],
        max_size_bytes: Optional[int],
    ) -> Optional[Dict[str, object]]:
        if extensions and file_path.suffix.lower() not in {ext.lower() for ext in extensions}:
            return None
        try:
            size = file_path.stat().st_size
        except OSError:
            return None
        if min_size_bytes is not None and size < min_size_bytes:
            return None
        if max_size_bytes is not None and size > max_size_bytes:
            return None

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/octet-stream"

        try:
            with file_path.open("rb") as f:
                data = f.read()
        except OSError:
            return None

        entropy = compute_entropy(data)
        printable = sum(32 <= b <= 126 for b in data)
        nulls = data.count(0)
        num_strings = sum(len(chunk) for chunk in bytes(data).split(b"\x00") if chunk)
        avg_string_length = float(num_strings) / max(1, len(bytes(data).split(b"\x00")))
        printable_ratio = clamp_ratio(printable / max(1, len(data)))
        null_ratio = clamp_ratio(nulls / max(1, len(data)))

        record_hash = hashlib.sha256(f"{file_path}:{size}".encode()).hexdigest()

        record = {
            "file_size_bytes": size,
            "file_extension": file_path.suffix.lower() or "<none>",
            "mime_type": mime_type,
            "creation_time_delta": max(0.0, time.time() - file_path.stat().st_ctime),
            "entropy": entropy,
            "num_strings": num_strings,
            "avg_string_length": avg_string_length,
            "printable_ratio": printable_ratio,
            "null_byte_ratio": null_ratio,
            "number_of_sections": 1,
            "has_executable_flag": file_path.suffix.lower() in {".exe", ".dll", ".bin"},
            "has_macros": file_path.suffix.lower() in {".docm", ".xlsm", ".pptm"},
            "imported_functions_count": 0,
            "suspicious_api_score": 0.0,
            "source_channel": self.context_defaults.get("source_channel", "filesystem"),
            "user_privilege_level": self.context_defaults.get("user_privilege_level", "unknown"),
            "access_time_category": self.context_defaults.get("access_time_category", "recent"),
            "class_label": "benign",
            "split": None,
            "record_id": record_hash,
        }
        return record


def normalize_class_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "class_label" not in df:
        df["class_label"] = "benign"
    df.loc[~df["class_label"].isin(DEFAULT_CLASSES), "class_label"] = "potentially_unwanted"
    return df
