from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_CLASSES, SynthConfig
from .utils import bounded_normal, clamp_ratio, weighted_choice


@dataclass
class ClassFeatureProfile:
    file_size_range: tuple
    entropy: tuple
    num_strings: tuple
    printable_ratio: tuple
    null_byte_ratio: tuple
    number_of_sections: tuple
    has_executable_prob: float
    has_macros_prob: float
    imported_functions_count: tuple
    suspicious_api_score: tuple
    extensions: List[str]
    mime_types: List[str]
    source_channels: List[str]
    privilege_levels: List[str]
    access_times: List[str]


DEFAULT_PROFILES: Dict[str, ClassFeatureProfile] = {
    "benign": ClassFeatureProfile(
        file_size_range=(10_000, 5_000_000),
        entropy=(3.5, 6.5),
        num_strings=(50, 4000),
        printable_ratio=(0.5, 0.95),
        null_byte_ratio=(0.0, 0.02),
        number_of_sections=(1, 4),
        has_executable_prob=0.05,
        has_macros_prob=0.05,
        imported_functions_count=(0, 20),
        suspicious_api_score=(0.0, 0.25),
        extensions=[".txt", ".log", ".cfg", ".csv"],
        mime_types=["text/plain", "text/csv", "application/json"],
        source_channels=["internal", "web"],
        privilege_levels=["standard", "elevated"],
        access_times=["recent", "archival"],
    ),
    "malware": ClassFeatureProfile(
        file_size_range=(20_000, 10_000_000),
        entropy=(6.0, 8.0),
        num_strings=(10, 800),
        printable_ratio=(0.1, 0.6),
        null_byte_ratio=(0.0, 0.15),
        number_of_sections=(3, 9),
        has_executable_prob=0.9,
        has_macros_prob=0.25,
        imported_functions_count=(20, 200),
        suspicious_api_score=(0.6, 1.0),
        extensions=[".exe", ".dll", ".bin", ".scr"],
        mime_types=["application/x-dosexec", "application/octet-stream"],
        source_channels=["email", "usb", "web"],
        privilege_levels=["elevated", "system"],
        access_times=["recent"],
    ),
    "potentially_unwanted": ClassFeatureProfile(
        file_size_range=(15_000, 7_000_000),
        entropy=(4.5, 7.0),
        num_strings=(200, 4000),
        printable_ratio=(0.35, 0.85),
        null_byte_ratio=(0.0, 0.08),
        number_of_sections=(2, 6),
        has_executable_prob=0.35,
        has_macros_prob=0.15,
        imported_functions_count=(10, 80),
        suspicious_api_score=(0.3, 0.75),
        extensions=[".exe", ".dll", ".msi", ".docm"],
        mime_types=["application/x-msdownload", "application/octet-stream"],
        source_channels=["web", "usb", "email"],
        privilege_levels=["standard", "elevated"],
        access_times=["recent", "during_off_hours"],
    ),
    "policy_violation": ClassFeatureProfile(
        file_size_range=(5_000, 3_000_000),
        entropy=(3.0, 6.5),
        num_strings=(100, 6000),
        printable_ratio=(0.45, 0.95),
        null_byte_ratio=(0.0, 0.05),
        number_of_sections=(1, 3),
        has_executable_prob=0.1,
        has_macros_prob=0.2,
        imported_functions_count=(0, 40),
        suspicious_api_score=(0.2, 0.6),
        extensions=[".docx", ".xlsx", ".pptx", ".pdf"],
        mime_types=["application/pdf", "application/vnd.ms-powerpoint", "application/vnd.ms-excel"],
        source_channels=["email", "internal"],
        privilege_levels=["standard"],
        access_times=["business_hours"],
    ),
    "confidential_suspected": ClassFeatureProfile(
        file_size_range=(50_000, 20_000_000),
        entropy=(4.0, 7.0),
        num_strings=(500, 10_000),
        printable_ratio=(0.55, 0.97),
        null_byte_ratio=(0.0, 0.03),
        number_of_sections=(1, 5),
        has_executable_prob=0.02,
        has_macros_prob=0.25,
        imported_functions_count=(0, 15),
        suspicious_api_score=(0.2, 0.5),
        extensions=[".docx", ".xlsx", ".pptx", ".pdf", ".zip"],
        mime_types=["application/pdf", "application/zip", "application/msword"],
        source_channels=["email", "internal", "cloud"],
        privilege_levels=["standard", "elevated"],
        access_times=["recent", "archival"],
    ),
}


class SyntheticDatasetGenerator:
    """Generate synthetic samples for multiple classes."""

    def __init__(self, profiles: Optional[Dict[str, ClassFeatureProfile]] = None) -> None:
        self.profiles = profiles or DEFAULT_PROFILES

    def generate(
        self,
        n_samples: int,
        class_distribution: Dict[str, float],
        noise_level: float = 0.05,
        seed: int = 7,
        hybrid_ratio: float = 0.0,
        hybrid_strength: float = 0.4,
        adversarial_ratio: float = 0.0,
        smoothing_factor: float = 0.25,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        counts = self._resolve_counts(n_samples, class_distribution)
        rows = []
        for label, count in counts.items():
            profile = self.profiles.get(label)
            if profile is None:
                continue
            rows.extend(self._generate_for_class(label, profile, count, noise_level, rng))
        df = pd.DataFrame(rows)
        if hybrid_ratio > 0:
            df = self._apply_hybrid(df, hybrid_ratio, hybrid_strength, rng)
        if adversarial_ratio > 0:
            df = self._apply_adversarial(df, adversarial_ratio, smoothing_factor, rng)
        df["record_id"] = [f"synth-{idx}" for idx in range(len(df))]
        return df

    def _resolve_counts(self, n_samples: int, distribution: Dict[str, float]) -> Dict[str, int]:
        if not distribution:
            distribution = {label: 1.0 / len(DEFAULT_CLASSES) for label in DEFAULT_CLASSES}
        total = sum(distribution.values())
        if total <= 0:
            raise ValueError("Class distribution must have positive weights")
        normalized = {label: weight / total for label, weight in distribution.items()}
        counts = {label: int(round(n_samples * weight)) for label, weight in normalized.items()}
        diff = n_samples - sum(counts.values())
        labels = list(distribution.keys())
        for i in range(abs(diff)):
            counts[labels[i % len(labels)]] += 1 if diff > 0 else -1
        return counts

    def _generate_for_class(
        self,
        label: str,
        profile: ClassFeatureProfile,
        count: int,
        noise_level: float,
        rng: np.random.Generator,
    ) -> List[Dict[str, object]]:
        sizes = rng.integers(profile.file_size_range[0], profile.file_size_range[1], size=count)
        entropy = rng.uniform(profile.entropy[0], profile.entropy[1], size=count)
        num_strings = rng.integers(profile.num_strings[0], profile.num_strings[1], size=count)
        printable = rng.uniform(profile.printable_ratio[0], profile.printable_ratio[1], size=count)
        null_ratio = rng.uniform(profile.null_byte_ratio[0], profile.null_byte_ratio[1], size=count)
        avg_string_length = bounded_normal(20, 5, 1, 120, count, rng)
        sections = rng.integers(profile.number_of_sections[0], profile.number_of_sections[1] + 1, size=count)
        imported = rng.integers(profile.imported_functions_count[0], profile.imported_functions_count[1] + 1, size=count)
        suspicious = rng.uniform(profile.suspicious_api_score[0], profile.suspicious_api_score[1], size=count)
        has_exec = rng.binomial(1, profile.has_executable_prob, size=count) > 0
        has_macros = rng.binomial(1, profile.has_macros_prob, size=count) > 0

        extensions = weighted_choice(profile.extensions, [1 / len(profile.extensions)] * len(profile.extensions), count, rng)
        mime_types = weighted_choice(profile.mime_types, [1 / len(profile.mime_types)] * len(profile.mime_types), count, rng)
        source_channel = weighted_choice(profile.source_channels, [1 / len(profile.source_channels)] * len(profile.source_channels), count, rng)
        privilege_levels = weighted_choice(profile.privilege_levels, [1 / len(profile.privilege_levels)] * len(profile.privilege_levels), count, rng)
        access_times = weighted_choice(profile.access_times, [1 / len(profile.access_times)] * len(profile.access_times), count, rng)

        rows = []
        for i in range(count):
            row = {
                "file_size_bytes": int(sizes[i] * (1 + rng.normal(0, noise_level))),
                "file_extension": extensions[i],
                "mime_type": mime_types[i],
                "creation_time_delta": float(abs(rng.normal(loc=86_400, scale=86_400))),
                "entropy": clamp_ratio(entropy[i]),
                "num_strings": int(num_strings[i]),
                "avg_string_length": float(avg_string_length[i]),
                "printable_ratio": clamp_ratio(printable[i]),
                "null_byte_ratio": clamp_ratio(null_ratio[i]),
                "number_of_sections": int(sections[i]),
                "has_executable_flag": bool(has_exec[i]),
                "has_macros": bool(has_macros[i]),
                "imported_functions_count": int(imported[i]),
                "suspicious_api_score": float(clamp_ratio(suspicious[i])),
                "source_channel": source_channel[i],
                "user_privilege_level": privilege_levels[i],
                "access_time_category": access_times[i],
                "class_label": label,
                "split": None,
            }
            rows.append(row)
        return rows

    def _apply_hybrid(
        self,
        df: pd.DataFrame,
        ratio: float,
        strength: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        if ratio <= 0:
            return df
        if "benign" not in df["class_label"].unique():
            return df
        n_rows = len(df)
        hybrid_count = int(n_rows * ratio)
        hybrid_indices = rng.choice(df.index, size=hybrid_count, replace=False)
        benign_pool = df[df["class_label"] == "benign"]
        benign_indices = rng.choice(benign_pool.index, size=hybrid_count, replace=True)
        numeric_cols = [
            "file_size_bytes",
            "entropy",
            "num_strings",
            "avg_string_length",
            "printable_ratio",
            "null_byte_ratio",
            "number_of_sections",
            "imported_functions_count",
            "suspicious_api_score",
        ]
        df = df.copy()
        for target_idx, benign_idx in zip(hybrid_indices, benign_indices):
            for col in numeric_cols:
                df.loc[target_idx, col] = float(
                    strength * df.loc[target_idx, col] + (1 - strength) * df.loc[benign_idx, col]
                )
            df.loc[target_idx, "has_executable_flag"] = bool(df.loc[target_idx, "has_executable_flag"])
            df.loc[target_idx, "has_macros"] = df.loc[target_idx, "has_macros"] or df.loc[benign_idx, "has_macros"]
        return df

    def _apply_adversarial(
        self,
        df: pd.DataFrame,
        ratio: float,
        smoothing_factor: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        if ratio <= 0:
            return df
        df = df.copy()
        distort_count = int(len(df) * ratio)
        indices = rng.choice(df.index, size=distort_count, replace=False)
        numeric_cols = [
            "entropy",
            "printable_ratio",
            "null_byte_ratio",
            "suspicious_api_score",
        ]
        for idx in indices:
            for col in numeric_cols:
                df.loc[idx, col] = float((1 - smoothing_factor) * df.loc[idx, col] + smoothing_factor * 0.5)
            if rng.random() < 0.5:
                df.loc[idx, "mime_type"] = "application/octet-stream"
            if rng.random() < 0.5:
                df.loc[idx, "source_channel"] = "internal"
        return df
