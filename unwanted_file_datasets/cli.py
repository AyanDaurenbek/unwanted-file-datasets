from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from .builder import DatasetBuilder
from .synthetic import SyntheticDatasetGenerator
from .collect import RealDataCollector
from .utils import save_dataframe


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def command_generate_synth(args: argparse.Namespace) -> None:
    config = load_yaml(Path(args.config))
    generator = SyntheticDatasetGenerator()
    df = generator.generate(
        n_samples=int(config.get("n_samples", 1000)),
        class_distribution=config.get("class_distribution", {}),
        noise_level=float(config.get("noise_level", 0.05)),
        seed=int(config.get("seed", 7)),
        hybrid_ratio=float(config.get("hybrid_ratio", 0.0)),
        hybrid_strength=float(config.get("hybrid_strength", 0.4)),
        adversarial_ratio=float(config.get("adversarial_ratio", 0.0)),
        smoothing_factor=float(config.get("smoothing_factor", 0.25)),
    )
    output_dir = Path(config.get("output_dir", "outputs/synthetic"))
    formats = config.get("export_formats", ("csv", "json", "parquet"))
    save_dataframe(df, output_dir, formats)


def command_collect_real(args: argparse.Namespace) -> None:
    config = load_yaml(Path(args.config))
    collector = RealDataCollector(context_defaults=config.get("context_defaults"))
    df = collector.collect(
        paths=[Path(p) for p in config.get("paths", [])],
        extensions=config.get("extensions"),
        min_size_bytes=config.get("min_size_bytes"),
        max_size_bytes=config.get("max_size_bytes"),
        exclude_paths=[Path(p) for p in config.get("exclude_paths", [])],
    )
    output_dir = Path(config.get("output_dir", "outputs/real"))
    formats = config.get("export_formats", ("csv", "json", "parquet"))
    save_dataframe(df, output_dir, formats)


def command_build_dataset(args: argparse.Namespace) -> None:
    config = load_yaml(Path(args.config))
    builder = DatasetBuilder()
    builder.build(
        real_sources=config.get("real_sources"),
        synth_config=config.get("synth_config"),
        split_config=config.get("split_config", {}),
        export_config=config.get("export_config", {}),
        seed=config.get("seed", 7),
        save_configs=bool(config.get("save_configs", True)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset builder for unwanted files")
    subparsers = parser.add_subparsers(dest="command")

    synth_parser = subparsers.add_parser("generate-synth", help="Generate synthetic dataset")
    synth_parser.add_argument("--config", required=True, help="Path to YAML configuration")
    synth_parser.set_defaults(func=command_generate_synth)

    collect_parser = subparsers.add_parser("collect-real", help="Collect real file features")
    collect_parser.add_argument("--config", required=True, help="Path to YAML configuration")
    collect_parser.set_defaults(func=command_collect_real)

    build_parser_cmd = subparsers.add_parser("build-dataset", help="Build final dataset")
    build_parser_cmd.add_argument("--config", required=True, help="Path to YAML configuration")
    build_parser_cmd.set_defaults(func=command_build_dataset)

    return parser


def main(argv: Any = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
