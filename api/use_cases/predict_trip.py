from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from api.infrastructure.model_registry import ModelRegistry
from api.repositories.prediction_repository import PredictionRepository


@dataclass(frozen=True)
class PredictionResult:
    prediction_id: int
    trip_duration: int
    model_version: str


class PredictTripUseCase:
    # Single-trip prediction orchestrates model loading, inference and persistence.
    def __init__(self, model_registry: ModelRegistry, prediction_repository: PredictionRepository):
        self._model_registry = model_registry
        self._prediction_repository = prediction_repository

    def execute(self, payload: dict, model_version: str | None = None) -> PredictionResult:
        model, metadata = self._model_registry.load_model(model_version)
        input_frame = pd.DataFrame([payload])
        trip_duration = int(model.predict(input_frame)[0])
        prediction_id = self._prediction_repository.save_prediction(
            payload,
            trip_duration,
            metadata.version,
        )
        return PredictionResult(
            prediction_id=prediction_id,
            trip_duration=trip_duration,
            model_version=metadata.version,
        )
