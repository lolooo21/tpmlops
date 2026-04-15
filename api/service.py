from __future__ import annotations

import sqlite3
from functools import lru_cache

import pandas as pd

from api.config import SETTINGS
from api.model_registry import ModelRegistry
from api.repository import build_prediction_repository


class PredictionService:
    # Service layer isolates I/O and inference from the FastAPI routes.
    def __init__(self, model_registry: ModelRegistry, repository):
        self._model_registry = model_registry
        self._repository = repository

    def predict(self, payload: dict, model_version: str | None = None) -> tuple[int, int, str]:
        # Single prediction reuses the batch pipeline to keep model selection consistent.
        prediction_rows, resolved_model_version = self.predict_batch([payload], model_version)
        prediction_id, trip_duration = prediction_rows[0]
        return prediction_id, trip_duration, resolved_model_version

    def predict_batch(
        self, payloads: list[dict], model_version: str | None = None
    ) -> tuple[list[tuple[int, int]], str]:
        # The model artifact expects a dataframe to reuse training-time feature logic.
        model, metadata = self._model_registry.load_model(model_version)
        input_frame = pd.DataFrame(payloads)
        predictions = model.predict(input_frame)

        persisted_predictions = []
        for payload, prediction in zip(payloads, predictions, strict=True):
            prediction = int(prediction)
            prediction_id = self._repository.save_prediction(
                payload,
                prediction,
                metadata.version,
            )
            persisted_predictions.append((prediction_id, prediction))

        return persisted_predictions, metadata.version

    def get_available_model_versions(self) -> list[str]:
        return self._model_registry.get_available_versions()

    def get_latest_model_version(self) -> str:
        return self._model_registry.get_metadata().version

    def get_random_test_row(self) -> tuple[dict, float]:
        # This endpoint is useful to inspect a realistic payload quickly.
        with sqlite3.connect(SETTINGS.db_path) as connection:
            data_test = pd.read_sql("SELECT * FROM test ORDER BY RANDOM() LIMIT 1", connection)

        y = float(data_test[SETTINGS.target_column].iloc[0])
        x = data_test.drop(columns=[SETTINGS.target_column]).iloc[0].to_dict()
        return x, y


@lru_cache
def load_prediction_service() -> PredictionService:
    # Cache the service so model resolution and repository wiring happen once per process.
    repository = build_prediction_repository()
    model_registry = ModelRegistry(
        latest_metadata_path=SETTINGS.model_metadata_path,
        versions_dir=SETTINGS.model_versions_dir,
        repository=repository,
    )
    return PredictionService(model_registry, repository)
