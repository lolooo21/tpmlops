from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from api.config import SETTINGS


EARTH_RADIUS_METERS = 6_371_000


def validate_nyc_coordinate(longitude: float, latitude: float, point_name: str) -> None:
    # Reject points that are outside the geographic area covered by the dataset.
    min_longitude = SETTINGS.longitude_range.min
    max_longitude = SETTINGS.longitude_range.max
    min_latitude = SETTINGS.latitude_range.min
    max_latitude = SETTINGS.latitude_range.max

    if not min_longitude <= longitude <= max_longitude:
        raise ValueError(
            f"{point_name}_longitude must be within the NYC bounding box "
            f"[{min_longitude}, {max_longitude}]."
        )

    if not min_latitude <= latitude <= max_latitude:
        raise ValueError(
            f"{point_name}_latitude must be within the NYC bounding box "
            f"[{min_latitude}, {max_latitude}]."
        )


def haversine_distance_meters(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
) -> float:
    # Haversine measures the straight-line distance on the Earth's surface
    # from two latitude/longitude pairs. Here we use it to reject trips that
    # are almost at the same point before calling the model.
    pickup_latitude = radians(pickup_latitude)
    pickup_longitude = radians(pickup_longitude)
    dropoff_latitude = radians(dropoff_latitude)
    dropoff_longitude = radians(dropoff_longitude)

    latitude_delta = dropoff_latitude - pickup_latitude
    longitude_delta = dropoff_longitude - pickup_longitude

    # Convert the angular gap between the two points into a surface distance.
    haversine = (
        sin(latitude_delta / 2) ** 2
        + cos(pickup_latitude) * cos(dropoff_latitude) * sin(longitude_delta / 2) ** 2
    )
    arc = 2 * asin(sqrt(haversine))
    return EARTH_RADIUS_METERS * arc


def validate_trip_coordinates(
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
) -> None:
    # Keep request validation focused on coordinates and minimum trip realism.
    validate_nyc_coordinate(pickup_longitude, pickup_latitude, "pickup")
    validate_nyc_coordinate(dropoff_longitude, dropoff_latitude, "dropoff")

    trip_distance = haversine_distance_meters(
        pickup_latitude=pickup_latitude,
        pickup_longitude=pickup_longitude,
        dropoff_latitude=dropoff_latitude,
        dropoff_longitude=dropoff_longitude,
    )
    if trip_distance <= SETTINGS.min_trip_distance_meters:
        raise ValueError(
            f"Haversine distance between pickup and dropoff must be greater than "
            f"{SETTINGS.min_trip_distance_meters} meters."
        )
