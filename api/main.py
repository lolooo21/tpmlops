from __future__ import annotations

from fastapi import Depends, FastAPI

from api.config import SETTINGS
from api.schemas import (
    HealthResponse,
    RandomTestResponse,
    TripPredictionRequest,
    TripPredictionResponse,
)
from api.service import PredictionService, load_prediction_service


# The FastAPI app only wires HTTP concerns to the service layer.
app = FastAPI(title=SETTINGS.title, version=SETTINGS.version)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Taxi Trip Duration API is running"}


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    # Lightweight readiness check that does not depend on the database.
    return HealthResponse(status="ok")


@app.post("/predict", response_model=TripPredictionResponse)
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
