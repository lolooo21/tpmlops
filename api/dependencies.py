from __future__ import annotations

from functools import lru_cache

from api.config import SETTINGS
from api.infrastructure.model_registry import ModelRegistry
from api.repositories.prediction_repository import build_prediction_repository
from api.repositories.test_data_repository import build_test_data_repository
from api.use_cases.get_random_test_sample import GetRandomTestSampleUseCase
from api.use_cases.predict_trip import PredictTripUseCase
from api.use_cases.predict_trip_batch import PredictTripBatchUseCase


@lru_cache
def get_model_registry() -> ModelRegistry:
    prediction_repository = build_prediction_repository()
    return ModelRegistry(
        latest_metadata_path=SETTINGS.model_metadata_path,
        versions_dir=SETTINGS.model_versions_dir,
        repository=prediction_repository,
    )


@lru_cache
def get_predict_trip_use_case() -> PredictTripUseCase:
    prediction_repository = build_prediction_repository()
    return PredictTripUseCase(
        model_registry=get_model_registry(),
        prediction_repository=prediction_repository,
    )


@lru_cache
def get_predict_trip_batch_use_case() -> PredictTripBatchUseCase:
    prediction_repository = build_prediction_repository()
    return PredictTripBatchUseCase(
        model_registry=get_model_registry(),
        prediction_repository=prediction_repository,
    )


@lru_cache
def get_random_test_sample_use_case() -> GetRandomTestSampleUseCase:
    return GetRandomTestSampleUseCase(test_data_repository=build_test_data_repository())
