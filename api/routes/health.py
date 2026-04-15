from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import get_model_registry
from api.schemas.health import HealthResponse, RootResponse

router = APIRouter()


def get_latest_available_model_version() -> str | None:
    # Keep the root endpoint informative even before the first training run.
    try:
        return get_model_registry().get_metadata().version
    except FileNotFoundError:
        return None


@router.get("/", response_model=RootResponse)
def root() -> RootResponse:
    from api.config import SETTINGS

    return RootResponse(
        message="Taxi Trip Duration API is running",
        nyc_bounding_box=SETTINGS.nyc_bounding_box_summary(),
        min_trip_distance_meters=SETTINGS.min_trip_distance_meters,
        latest_model_version=get_latest_available_model_version(),
    )


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")
