from __future__ import annotations

from dataclasses import dataclass

import common


@dataclass(frozen=True)
class ApiSettings:
    # API settings stay close to the application layer while reusing
    # the project-wide configuration resolved in common.py.
    title: str = "Taxi Trip Duration API"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = common.CONFIG["paths"]["model_custom_path"]
    db_path: str = common.CONFIG["paths"]["db_path"]
    target_column: str = common.CONFIG["dataset"]["target_column"]


# Single settings instance imported by the API modules.
SETTINGS = ApiSettings()
