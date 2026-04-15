from __future__ import annotations

import pickle
from pathlib import Path

from api.exceptions import ModelVersionNotFoundError
from api.model_metadata import ModelMetadata, load_model_metadata
from api.repository import PredictionRepository
from model.inference.custom_model import TaxiTripDurationModel


class ModelRegistry:
    # Registry resolves model versions and caches loaded artifacts in memory.
    def __init__(self, latest_metadata_path: str, versions_dir: str, repository: PredictionRepository):
        self._latest_metadata_path = Path(latest_metadata_path).resolve()
        self._versions_dir = Path(versions_dir).resolve()
        self._repository = repository
        self._models_by_version: dict[str, TaxiTripDurationModel] = {}

    def get_metadata(self, model_version: str | None = None) -> ModelMetadata:
        # Default to the latest available version when the caller does not specify one.
        metadata_by_version = self._load_metadata_by_version()
        if model_version is None:
            return max(
                metadata_by_version.values(),
                key=lambda metadata: (metadata.created_at, metadata.version),
            )

        if model_version not in metadata_by_version:
            raise ModelVersionNotFoundError(model_version, sorted(metadata_by_version))
        return metadata_by_version[model_version]

    def get_available_versions(self) -> list[str]:
        return sorted(self._load_metadata_by_version())

    def load_model(self, model_version: str | None = None) -> tuple[TaxiTripDurationModel, ModelMetadata]:
        # Load the requested model once and reuse it across requests.
        metadata = self.get_metadata(model_version)
        cached_model = self._models_by_version.get(metadata.version)
        if cached_model is None:
            with open(metadata.path, "rb") as file:
                cached_model = pickle.load(file)

            if not isinstance(cached_model, TaxiTripDurationModel):
                raise TypeError(
                    "The persisted model must be a TaxiTripDurationModel. "
                    "Train it again with `python -m model.training.train_custom_model`."
                )

            self._models_by_version[metadata.version] = cached_model
            self._repository.register_model_metadata(metadata)

        return cached_model, metadata

    def _load_metadata_by_version(self) -> dict[str, ModelMetadata]:
        # Versioned metadata files are the source of truth for available models.
        metadata_by_version: dict[str, ModelMetadata] = {}

        if self._versions_dir.exists():
            for metadata_file in self._versions_dir.glob("*.metadata.json"):
                metadata = load_model_metadata(str(metadata_file))
                metadata_by_version[metadata.version] = metadata

        if self._latest_metadata_path.exists():
            latest_metadata = load_model_metadata(str(self._latest_metadata_path))
            metadata_by_version.setdefault(latest_metadata.version, latest_metadata)

        if not metadata_by_version:
            raise FileNotFoundError(
                "No model metadata found. Train a model with "
                "`python -m model.training.train_custom_model`."
            )

        return metadata_by_version
