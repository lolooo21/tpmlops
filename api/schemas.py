from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TripPredictionRequest(BaseModel):
    # This schema mirrors the raw columns expected by the inference layer.
    vendor_id: int = Field(..., ge=1)
    pickup_datetime: datetime
    passenger_count: int = Field(..., ge=0)
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str = Field(..., min_length=1, max_length=1)


class TripPredictionResponse(BaseModel):
    # Response stays intentionally minimal: one request, one prediction.
    prediction_id: int
    trip_duration: int


class HealthResponse(BaseModel):
    # Operational endpoint used by manual checks or orchestration tools.
    status: str


class RandomTestResponse(BaseModel):
    # Debug endpoint payload exposing one database row and its target value.
    x: dict
    y: float
