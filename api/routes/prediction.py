from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.dependencies import (
    get_predict_trip_batch_use_case,
    get_predict_trip_use_case,
    get_random_test_sample_use_case,
)
from api.schemas.errors import ModelVersionErrorResponse, ValidationErrorResponse
from api.schemas.health import RandomTestResponse
from api.schemas.prediction import (
    BatchPredictionItem,
    BatchTripPredictionRequest,
    BatchTripPredictionResponse,
    TripPredictionRequest,
    TripPredictionResponse,
)
from api.use_cases.get_random_test_sample import GetRandomTestSampleUseCase
from api.use_cases.predict_trip import PredictTripUseCase
from api.use_cases.predict_trip_batch import PredictTripBatchUseCase

router = APIRouter()


@router.post(
    "/predict",
    response_model=TripPredictionResponse,
    responses={
        404: {
            "model": ModelVersionErrorResponse,
            "description": "Requested model version was not found.",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "Payload rejected by input validation.",
        },
    },
    summary="Predict one NYC taxi trip duration",
    description=(
        "Coordinates must stay inside the configured NYC bounding box and the "
        "Haversine distance must be greater than the configured minimum distance. "
        "Set model_version to target a specific registered model version."
    ),
)
def predict(
    request: TripPredictionRequest,
    model_version: str | None = Query(
        default=None,
        description="Optional model version. If omitted, the latest available version is used.",
    ),
    use_case: PredictTripUseCase = Depends(get_predict_trip_use_case),
) -> TripPredictionResponse:
    result = use_case.execute(request.model_dump(), model_version)
    return TripPredictionResponse(
        prediction_id=result.prediction_id,
        trip_duration=result.trip_duration,
        model_version=result.model_version,
    )


@router.post(
    "/predict_batch",
    response_model=BatchTripPredictionResponse,
    responses={
        404: {
            "model": ModelVersionErrorResponse,
            "description": "Requested model version was not found.",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "Payload rejected by input validation.",
        },
    },
    summary="Predict multiple NYC taxi trip durations",
    description=(
        "Batch prediction endpoint using one resolved model version for all trips. "
        "If model_version is omitted, the latest available version is used."
    ),
)
def predict_batch(
    request: BatchTripPredictionRequest,
    model_version: str | None = Query(
        default=None,
        description="Optional model version. If omitted, the latest available version is used.",
    ),
    use_case: PredictTripBatchUseCase = Depends(get_predict_trip_batch_use_case),
) -> BatchTripPredictionResponse:
    result = use_case.execute([trip.model_dump() for trip in request.trips], model_version)
    return BatchTripPredictionResponse(
        model_version=result.model_version,
        predictions=[
            BatchPredictionItem(
                prediction_id=item.prediction_id,
                trip_duration=item.trip_duration,
            )
            for item in result.predictions
        ],
    )


@router.get("/test/random", response_model=RandomTestResponse)
def random_test_sample(
    use_case: GetRandomTestSampleUseCase = Depends(get_random_test_sample_use_case),
) -> RandomTestResponse:
    x, y = use_case.execute()
    return RandomTestResponse(x=x, y=y)
