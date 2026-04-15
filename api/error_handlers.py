from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.config import SETTINGS
from api.exceptions import ModelVersionNotFoundError
from api.schemas.errors import ModelVersionErrorResponse, ValidationErrorResponse


async def handle_request_validation_error(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    # Return validation details with the configured geographic constraints.
    details = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"] if part != "body")
        details.append(
            {
                "field": location or "body",
                "message": item["msg"],
            }
        )

    return JSONResponse(
        status_code=422,
        content=ValidationErrorResponse(
            message=(
                "Invalid request payload. Check the coordinate ranges and the "
                "minimum distance before retrying."
            ),
            bounding_box={
                "longitude": [SETTINGS.longitude_range.min, SETTINGS.longitude_range.max],
                "latitude": [SETTINGS.latitude_range.min, SETTINGS.latitude_range.max],
            },
            min_trip_distance_meters=SETTINGS.min_trip_distance_meters,
            errors=details,
        ).model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    # Keep exception wiring in one place to simplify api.main.
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)
    app.add_exception_handler(ModelVersionNotFoundError, handle_model_version_not_found)


async def handle_model_version_not_found(
    request: Request, error: ModelVersionNotFoundError
) -> JSONResponse:
    # Make invalid model version requests explicit for API consumers.
    return JSONResponse(
        status_code=404,
        content=ModelVersionErrorResponse(
            message="Requested model version is not available.",
            requested_model_version=error.requested_version,
            available_model_versions=error.available_versions,
        ).model_dump(),
    )
