from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from api.config import SETTINGS
from api.validation import validate_trip_coordinates


class TripPredictionRequest(BaseModel):
    # This schema mirrors the raw columns expected by the inference layer.
    vendor_id: int = Field(..., ge=1)
    pickup_datetime: datetime
    passenger_count: int = Field(..., ge=0)
    pickup_longitude: float = Field(
        ...,
        description=(
            "Pickup longitude inside the NYC bounding box: "
            f"[{SETTINGS.longitude_range.min}, {SETTINGS.longitude_range.max}]."
        ),
    )
    pickup_latitude: float = Field(
        ...,
        description=(
            "Pickup latitude inside the NYC bounding box: "
            f"[{SETTINGS.latitude_range.min}, {SETTINGS.latitude_range.max}]."
        ),
    )
    dropoff_longitude: float = Field(
        ...,
        description=(
            "Dropoff longitude inside the NYC bounding box: "
            f"[{SETTINGS.longitude_range.min}, {SETTINGS.longitude_range.max}]."
        ),
    )
    dropoff_latitude: float = Field(
        ...,
        description=(
            "Dropoff latitude inside the NYC bounding box: "
            f"[{SETTINGS.latitude_range.min}, {SETTINGS.latitude_range.max}]."
        ),
    )
    store_and_fwd_flag: str = Field(..., min_length=1, max_length=1)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "TripPredictionRequest":
        # Centralize geo checks so the route only receives coherent trips.
        validate_trip_coordinates(
            pickup_longitude=self.pickup_longitude,
            pickup_latitude=self.pickup_latitude,
            dropoff_longitude=self.dropoff_longitude,
            dropoff_latitude=self.dropoff_latitude,
        )
        return self


class TripPredictionResponse(BaseModel):
    # Response stays intentionally minimal: one request, one prediction.
    prediction_id: int
    trip_duration: int


class HealthResponse(BaseModel):
    # Operational endpoint used by manual checks or orchestration tools.
    status: str


class RootResponse(BaseModel):
    # Root endpoint exposes quick usage hints for manual API checks.
    message: str
    nyc_bounding_box: str
    min_trip_distance_meters: int


class RandomTestResponse(BaseModel):
    # Debug endpoint payload exposing one database row and its target value.
    x: dict
    y: float


class ValidationErrorResponse(BaseModel):
    # Keep 422 responses explicit so API users know how to fix their payload.
    message: str
    bounding_box: dict[str, list[float]]
    min_trip_distance_meters: int
    errors: list[dict[str, str]]
