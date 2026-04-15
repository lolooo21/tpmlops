from __future__ import annotations

from dataclasses import dataclass

import common


@dataclass(frozen=True)
class CoordinateRange:
    min: float
    max: float


@dataclass(frozen=True)
class ApiSettings:
    # API settings stay close to the application layer while reusing
    # the project-wide configuration resolved in common.py.
    title: str = "Taxi Trip Duration API"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = common.CONFIG["paths"]["model_custom_path"]
    model_metadata_path: str = common.CONFIG["paths"]["model_custom_metadata_path"]
    model_versions_dir: str = common.CONFIG["paths"]["model_custom_versions_dir"]
    db_path: str = common.CONFIG["paths"]["db_path"]
    target_column: str = common.CONFIG["dataset"]["target_column"]
    min_trip_distance_meters: int = common.CONFIG["api"]["validation"]["min_trip_distance_meters"]
    longitude_range: CoordinateRange = CoordinateRange(
        min=common.CONFIG["api"]["validation"]["nyc_bounding_box"]["longitude"]["min"],
        max=common.CONFIG["api"]["validation"]["nyc_bounding_box"]["longitude"]["max"],
    )
    latitude_range: CoordinateRange = CoordinateRange(
        min=common.CONFIG["api"]["validation"]["nyc_bounding_box"]["latitude"]["min"],
        max=common.CONFIG["api"]["validation"]["nyc_bounding_box"]["latitude"]["max"],
    )

    def nyc_bounding_box_summary(self) -> str:
        return (
            f"longitude in [{self.longitude_range.min}, {self.longitude_range.max}], "
            f"latitude in [{self.latitude_range.min}, {self.latitude_range.max}]"
        )


# Single settings instance imported by the API modules.
SETTINGS = ApiSettings()
