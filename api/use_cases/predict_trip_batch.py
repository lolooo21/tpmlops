from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from api.infrastructure.model_registry import ModelRegistry
from api.repositories.prediction_repository import PredictionRepository


@dataclass(frozen=True)
class BatchPredictionItemResult:
    prediction_id: int
    trip_duration: int


@dataclass(frozen=True)
class BatchPredictionResult:
    model_version: str
    predictions: list[BatchPredictionItemResult]


class PredictTripBatchUseCase:
    # Batch prediction shares one resolved model version across all items.
    def __init__(self, model_registry: ModelRegistry, prediction_repository: PredictionRepository):
        self._model_registry = model_registry
        self._prediction_repository = prediction_repository

    def execute(
        self, payloads: list[dict], model_version: str | None = None
    ) -> BatchPredictionResult:
        model, metadata = self._model_registry.load_model(model_version)
        input_frame = pd.DataFrame(payloads)
        raw_predictions = model.predict(input_frame)

        predictions: list[BatchPredictionItemResult] = []
        for payload, raw_prediction in zip(payloads, raw_predictions, strict=True):
            trip_duration = int(raw_prediction)
            prediction_id = self._prediction_repository.save_prediction(
                payload,
                trip_duration,
                metadata.version,
            )
            predictions.append(
                BatchPredictionItemResult(
                    prediction_id=prediction_id,
                    trip_duration=trip_duration,
                )
            )

        return BatchPredictionResult(model_version=metadata.version, predictions=predictions)
