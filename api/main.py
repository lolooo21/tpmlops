from __future__ import annotations

from fastapi import FastAPI

from api.config import SETTINGS
from api.error_handlers import register_exception_handlers
from api.routes.health import router as health_router
from api.routes.prediction import router as prediction_router


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
app.include_router(health_router)
app.include_router(prediction_router)
