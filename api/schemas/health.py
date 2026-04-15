from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    # Operational endpoint used by manual checks or orchestration tools.
    status: str


class RootResponse(BaseModel):
    # Root endpoint exposes quick usage hints for manual API checks.
    message: str
    nyc_bounding_box: str
    min_trip_distance_meters: int
    latest_model_version: str | None


class RandomTestResponse(BaseModel):
    # Debug endpoint payload exposing one database row and its target value.
    x: dict
    y: float
