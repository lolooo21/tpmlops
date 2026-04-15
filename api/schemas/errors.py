from __future__ import annotations

from pydantic import BaseModel


class ValidationErrorResponse(BaseModel):
    # Keep 422 responses explicit so API users know how to fix their payload.
    message: str
    bounding_box: dict[str, list[float]]
    min_trip_distance_meters: int
    errors: list[dict[str, str]]


class ModelVersionErrorResponse(BaseModel):
    # Explain model selection failures with the available versions.
    message: str
    requested_model_version: str
    available_model_versions: list[str]
