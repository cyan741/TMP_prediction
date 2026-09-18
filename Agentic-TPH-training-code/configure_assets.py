#!/usr/bin/env python3
"""Register a downloaded base model for training and inference without copying weights."""

import argparse
import json
from pathlib import Path


def register_model(asset_root: Path, model_id: str, relative_root: Path) -> Path:
    """Register a local model directory in manifest.json and return the manifest path.

    The directory must contain the model weights, configuration and tokenizer.
    relative_root is relative to asset_root so the asset tree can be relocated.
    """
    if relative_root.is_absolute() or ".." in relative_root.parts:
        raise ValueError("relative-root must remain inside asset-root")
    model_root = asset_root / relative_root
    if not model_root.is_dir():
        raise FileNotFoundError(model_root)

    manifest_path = asset_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "models": {}}
    if model_id in manifest["models"]:
        raise FileExistsError(f"Model already registered: {model_id}")

    files = [
        {"path": str(path.relative_to(model_root)), "size": path.stat().st_size}
        for path in sorted(model_root.rglob("*"))
        if path.is_file()
    ]
    if not files:
        raise ValueError("Model directory is empty")
    manifest["models"][model_id] = {"relative_root": str(relative_root), "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for local model registration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        required=True,
        help="Asset root directory in which manifest.json is read and written",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model name used by training and inference, such as esm2-150M",
    )
    parser.add_argument(
        "--relative-root",
        type=Path,
        required=True,
        help="Model directory relative to asset-root, such as models/esm2-150M",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Register assets when invoked as a script; importing this module has no side effects."""
    args = build_parser().parse_args(argv)
    print(register_model(args.asset_root, args.model_id, args.relative_root))


if __name__ == "__main__":
    main()
