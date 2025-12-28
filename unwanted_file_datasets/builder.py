from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .collect import RealDataCollector, normalize_class_labels
from .config import BuildConfig, ExportConfig, RealSourceConfig, SynthConfig
from .synthetic import SyntheticDatasetGenerator
from .utils import describe_numeric, save_dataframe, save_json, save_yaml, stratified_split


class DatasetBuilder:
    def __init__(
        self,
        generator: Optional[SyntheticDatasetGenerator] = None,
        collector: Optional[RealDataCollector] = None,
    ) -> None:
        self.generator = generator or SyntheticDatasetGenerator()
        self.collector = collector or RealDataCollector()

    def build(
        self,
        real_sources: Optional[Dict],
        synth_config: Optional[Dict],
        split_config: Dict,
        export_config: Dict,
        seed: int = 7,
        save_configs: bool = True,
    ) -> pd.DataFrame:
        real_df = self._collect_real(real_sources) if real_sources else pd.DataFrame()
        synth_df = self._generate_synth(synth_config) if synth_config else pd.DataFrame()
        frames = [df for df in (real_df, synth_df) if not df.empty]
        if not frames:
            raise ValueError("No data sources provided")
        combined = pd.concat(frames, ignore_index=True)
        combined = normalize_class_labels(combined)
        combined = stratified_split(
            combined,
            (
                float(split_config.get("train", 0.7)),
                float(split_config.get("val", 0.15)),
                float(split_config.get("test", 0.15)),
            ),
            seed,
        )
        export = ExportConfig(output_dir=Path(export_config["output_dir"]), formats=export_config.get("formats", ("csv", "json", "parquet")))
        save_dataframe(combined, export.output_dir, export.formats)
        if save_configs:
            self._persist_configs(real_sources, synth_config, split_config, export_config, seed, export.output_dir)
        self._write_dataset_card(combined, export.output_dir, seed)
        return combined

    def _collect_real(self, config_dict: Dict) -> pd.DataFrame:
        config = RealSourceConfig(
            paths=[Path(p) for p in config_dict.get("paths", [])],
            extensions=config_dict.get("extensions"),
            min_size_bytes=config_dict.get("min_size_bytes"),
            max_size_bytes=config_dict.get("max_size_bytes"),
            exclude_paths=[Path(p) for p in config_dict.get("exclude_paths", [])],
            context_defaults=config_dict.get("context_defaults"),
        )
        collector = RealDataCollector(context_defaults=config.context_defaults)
        return collector.collect_from_config(config)

    def _generate_synth(self, config_dict: Dict) -> pd.DataFrame:
        config = SynthConfig(
            n_samples=int(config_dict["n_samples"]),
            class_distribution=config_dict.get("class_distribution", {}),
            noise_level=float(config_dict.get("noise_level", 0.05)),
            hybrid_ratio=float(config_dict.get("hybrid_ratio", 0.0)),
            hybrid_strength=float(config_dict.get("hybrid_strength", 0.4)),
            adversarial_ratio=float(config_dict.get("adversarial_ratio", 0.0)),
            smoothing_factor=float(config_dict.get("smoothing_factor", 0.25)),
        )
        return self.generator.generate(
            n_samples=config.n_samples,
            class_distribution=config.class_distribution,
            noise_level=config.noise_level,
            seed=config_dict.get("seed", 7),
            hybrid_ratio=config.hybrid_ratio,
            hybrid_strength=config.hybrid_strength,
            adversarial_ratio=config.adversarial_ratio,
            smoothing_factor=config.smoothing_factor,
        )

    def _persist_configs(
        self,
        real_sources: Optional[Dict],
        synth_config: Optional[Dict],
        split_config: Dict,
        export_config: Dict,
        seed: int,
        output_dir: Path,
    ) -> None:
        payload = {
            "real_sources": real_sources or {},
            "synth_config": synth_config or {},
            "split_config": split_config,
            "export_config": export_config,
            "seed": seed,
        }
        save_yaml(payload, output_dir / "generation_config.yaml")

    def _write_dataset_card(self, df: pd.DataFrame, output_dir: Path, seed: int) -> None:
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
        class_sizes = df["class_label"].value_counts().to_dict()
        card = {
            "dataset_card_version": 1,
            "seed": seed,
            "num_records": int(len(df)),
            "class_balance": {k: int(v) for k, v in class_sizes.items()},
            "splits": df["split"].value_counts().to_dict(),
            "numeric_stats": describe_numeric(df, numeric_cols),
            "feature_schema": list(df.columns),
        }
        save_json(card, output_dir / "dataset_card.json")
