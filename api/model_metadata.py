from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ModelMetadata:
    # Keep model tracing explicit and independent from the prediction flow.
    version: str
    path: str
    created_at: str


def build_model_metadata(model_path: str, model_version: str) -> ModelMetadata:
    # Build a traceable metadata payload for the newly persisted model artifact.
    artifact_path = Path(model_path).resolve()
    created_at = datetime.fromtimestamp(artifact_path.stat().st_mtime, tz=timezone.utc).isoformat()
    return ModelMetadata(
        version=model_version,
        path=str(artifact_path),
        created_at=created_at,
    )


def load_model_metadata(metadata_path: str) -> ModelMetadata:
    # The API reads model identity from the metadata artifact produced at training time.
    metadata_file = Path(metadata_path).resolve()
    with metadata_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return ModelMetadata(**payload)


def save_model_metadata(metadata: ModelMetadata, metadata_path: str) -> None:
    # Persist model identity separately from config because it changes per training run.
    metadata_file = Path(metadata_path).resolve()
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with metadata_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "version": metadata.version,
                "path": metadata.path,
                "created_at": metadata.created_at,
            },
            file,
            indent=2,
        )
