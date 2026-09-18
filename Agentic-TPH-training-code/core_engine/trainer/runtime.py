from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol


class AssetConfig(Protocol):
    """Paths required by the local model asset loader."""

    asset_root: Path

    @property
    def manifest_path(self) -> Path:
        """Return the path to the model asset manifest."""
        ...


class TrainerPreflightError(RuntimeError):
    """A structured, fail-closed environment or data-contract failure."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


def load_asset_manifest(config: AssetConfig) -> dict[str, Any]:
    path = config.manifest_path
    if not path.is_file():
        raise TrainerPreflightError(
            "ASSET_MANIFEST_MISSING",
            "Trainer asset manifest does not exist",
            path=str(path),
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainerPreflightError(
            "ASSET_MANIFEST_INVALID",
            "Trainer asset manifest cannot be read",
            path=str(path),
            error=f"{type(exc).__name__}: {exc}",
        ) from exc
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("models"), dict
    ):
        raise TrainerPreflightError(
            "ASSET_MANIFEST_INVALID",
            "Trainer asset manifest schema is unsupported",
            path=str(path),
        )
    return payload


def validate_model_asset(
    config: AssetConfig,
    model_id: str,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = load_asset_manifest(config) if manifest is None else manifest
    raw = payload["models"].get(model_id)
    if not isinstance(raw, dict):
        raise TrainerPreflightError(
            "MODEL_ASSET_UNDECLARED",
            "Requested model is absent from the Trainer asset manifest",
            model_id=model_id,
        )
    relative_root = Path(str(raw.get("relative_root", "")))
    if relative_root.is_absolute() or ".." in relative_root.parts:
        raise TrainerPreflightError(
            "MODEL_ASSET_ROOT_INVALID",
            "Model asset root must stay inside TRAINER_ASSET_ROOT",
            model_id=model_id,
            relative_root=str(relative_root),
        )
    root = (config.asset_root / relative_root).resolve()
    asset_root = config.asset_root.resolve()
    if root != asset_root and asset_root not in root.parents:
        raise TrainerPreflightError(
            "MODEL_ASSET_ROOT_INVALID",
            "Model asset root escapes TRAINER_ASSET_ROOT",
            model_id=model_id,
            root=str(root),
        )
    missing: list[str] = []
    size_mismatch: list[dict[str, Any]] = []
    for item in raw.get("files", []):
        if not isinstance(item, dict) or not str(item.get("path", "")):
            raise TrainerPreflightError(
                "ASSET_MANIFEST_INVALID",
                "Model file entry is invalid",
                model_id=model_id,
            )
        relative_file = Path(str(item["path"]))
        if relative_file.is_absolute() or ".." in relative_file.parts:
            raise TrainerPreflightError(
                "MODEL_ASSET_FILE_INVALID",
                "Model file path must be relative",
                model_id=model_id,
                path=str(relative_file),
            )
        path = root / relative_file
        if not path.is_file():
            missing.append(str(relative_file))
            continue
        expected = int(item.get("size", -1))
        actual = path.stat().st_size
        if expected < 0 or actual != expected:
            size_mismatch.append(
                {"path": str(relative_file), "expected": expected, "actual": actual}
            )
    if missing or size_mismatch:
        raise TrainerPreflightError(
            "MODEL_ASSET_INCOMPLETE",
            "Requested model assets are incomplete",
            model_id=model_id,
            root=str(root),
            missing_files=missing,
            size_mismatch=size_mismatch,
        )
    result = dict(raw)
    result["root"] = str(root)
    return result
