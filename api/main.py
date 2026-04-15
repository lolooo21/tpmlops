from __future__ import annotations

from fastapi import Depends, FastAPI, Query

from api.config import SETTINGS
from api.error_handlers import register_exception_handlers
from api.schemas import (
    BatchPredictionItem,
    BatchTripPredictionRequest,
    BatchTripPredictionResponse,
    HealthResponse,
    ModelVersionErrorResponse,
    RandomTestResponse,
    RootResponse,
    TripPredictionRequest,
    TripPredictionResponse,
    ValidationErrorResponse,
)
from api.service import PredictionService, load_prediction_service


def create_app() -> FastAPI:
    # Build the FastAPI application and register its cross-cutting concerns.
    app = FastAPI(
        title=SETTINGS.title,
        version=SETTINGS.version,
        description=(
            "API for NYC taxi trip duration prediction.\n\n"
            "Input coordinates must stay inside the NYC bounding box: "
            f"{SETTINGS.nyc_bounding_box_summary()}.\n"
            f"The Haversine distance between pickup and dropoff must be greater than "
            f"{SETTINGS.min_trip_distance_meters} meters."
        ),
    )
    register_exception_handlers(app)
    return app


app = create_app()


def get_latest_available_model_version() -> str | None:
    # Keep the root endpoint informative even before the first training run.
    try:
        return load_prediction_service().get_latest_model_version()
    except FileNotFoundError:
        return None


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(
        message="Taxi Trip Duration API is running",
        nyc_bounding_box=SETTINGS.nyc_bounding_box_summary(),
        min_trip_distance_meters=SETTINGS.min_trip_distance_meters,
        latest_model_version=get_latest_available_model_version(),
    )


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    # Lightweight readiness check that does not depend on the database.
    return HealthResponse(status="ok")


@app.post(
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
        }
    },
    summary="Predict one NYC taxi trip duration",
    description=(
        "Coordinates must stay inside the configured NYC bounding box "
        f"({SETTINGS.nyc_bounding_box_summary()}) and the Haversine distance "
        f"must be greater than {SETTINGS.min_trip_distance_meters} meters. "
        "Set model_version to target a specific registered model version."
    ),
)
def predict(
    request: TripPredictionRequest,
    model_version: str | None = Query(
        default=None,
        description="Optional model version. If omitted, the latest available version is used.",
    ),
    service: PredictionService = Depends(load_prediction_service),
) -> TripPredictionResponse:
    # Request validation is handled by Pydantic before the service is called.
    prediction_id, prediction, resolved_model_version = service.predict(
        request.model_dump(),
        model_version,
    )
    return TripPredictionResponse(
        prediction_id=prediction_id,
        trip_duration=prediction,
        model_version=resolved_model_version,
    )


@app.post(
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
    service: PredictionService = Depends(load_prediction_service),
) -> BatchTripPredictionResponse:
    prediction_rows, resolved_model_version = service.predict_batch(
        [trip.model_dump() for trip in request.trips],
        model_version,
    )
    return BatchTripPredictionResponse(
        model_version=resolved_model_version,
        predictions=[
            BatchPredictionItem(prediction_id=prediction_id, trip_duration=trip_duration)
            for prediction_id, trip_duration in prediction_rows
        ],
    )


@app.get("/test/random", response_model=RandomTestResponse)
def random_test_sample(
    service: PredictionService = Depends(load_prediction_service),
) -> RandomTestResponse:
    # Exposes one real test sample to simplify manual API validation.
    x, y = service.get_random_test_row()
    return RandomTestResponse(x=x, y=y)
