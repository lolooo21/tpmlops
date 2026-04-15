from __future__ import annotations

from fastapi import Depends, FastAPI

from api.config import SETTINGS
from api.error_handlers import register_exception_handlers
from api.schemas import (
    HealthResponse,
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


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(
        message="Taxi Trip Duration API is running",
        nyc_bounding_box=SETTINGS.nyc_bounding_box_summary(),
        min_trip_distance_meters=SETTINGS.min_trip_distance_meters,
    )


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    # Lightweight readiness check that does not depend on the database.
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=TripPredictionResponse,
    responses={
        422: {
            "model": ValidationErrorResponse,
            "description": "Payload rejected by input validation.",
        }
    },
    summary="Predict one NYC taxi trip duration",
    description=(
        "Coordinates must stay inside the configured NYC bounding box "
        f"({SETTINGS.nyc_bounding_box_summary()}) and the Haversine distance "
        f"must be greater than {SETTINGS.min_trip_distance_meters} meters."
    ),
)
def predict(
    request: TripPredictionRequest,
    service: PredictionService = Depends(load_prediction_service),
) -> TripPredictionResponse:
    # Request validation is handled by Pydantic before the service is called.
    prediction_id, prediction = service.predict(request.model_dump())
    return TripPredictionResponse(prediction_id=prediction_id, trip_duration=prediction)


@app.get("/test/random", response_model=RandomTestResponse)
def random_test_sample(
    service: PredictionService = Depends(load_prediction_service),
) -> RandomTestResponse:
    # Exposes one real test sample to simplify manual API validation.
    x, y = service.get_random_test_row()
    return RandomTestResponse(x=x, y=y)
