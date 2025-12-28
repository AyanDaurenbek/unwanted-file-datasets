from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class RealSourceConfig:
    paths: List[Path]
    extensions: Optional[List[str]] = None
    min_size_bytes: Optional[int] = None
    max_size_bytes: Optional[int] = None
    exclude_paths: Optional[List[Path]] = None
    context_defaults: Optional[Dict[str, str]] = None


@dataclass
class SynthConfig:
    n_samples: int
    class_distribution: Dict[str, float]
    noise_level: float = 0.05
    hybrid_ratio: float = 0.0
    hybrid_strength: float = 0.4
    adversarial_ratio: float = 0.0
    smoothing_factor: float = 0.25


@dataclass
class SplitConfig:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")


@dataclass
class ExportConfig:
    output_dir: Path
    formats: Iterable[str] = field(default_factory=lambda: ("csv", "json", "parquet"))


@dataclass
class BuildConfig:
    real_sources: Optional[RealSourceConfig]
    synth_config: Optional[SynthConfig]
    split_config: SplitConfig
    export_config: ExportConfig
    seed: int = 7
    save_configs: bool = True


DEFAULT_CLASSES = [
    "benign",
    "malware",
    "potentially_unwanted",
    "policy_violation",
    "confidential_suspected",
]
