from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from api.config import SETTINGS


class PredictionRepository:
    # Repository isolates SQLite concerns from the service and the routes.
    def __init__(self, db_path: str):
        self._db_path = db_path

    def initialize(self) -> None:
        with sqlite3.connect(self._db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    vendor_id INTEGER NOT NULL,
                    pickup_datetime TEXT NOT NULL,
                    passenger_count INTEGER NOT NULL,
                    pickup_longitude REAL NOT NULL,
                    pickup_latitude REAL NOT NULL,
                    dropoff_longitude REAL NOT NULL,
                    dropoff_latitude REAL NOT NULL,
                    store_and_fwd_flag TEXT NOT NULL,
                    prediction INTEGER NOT NULL
                )
                """
            )
            connection.commit()

    def save_prediction(self, payload: dict, prediction: int) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        pickup_datetime = payload["pickup_datetime"]
        if hasattr(pickup_datetime, "isoformat"):
            pickup_datetime = pickup_datetime.isoformat()

        with sqlite3.connect(self._db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (
                    created_at,
                    vendor_id,
                    pickup_datetime,
                    passenger_count,
                    pickup_longitude,
                    pickup_latitude,
                    dropoff_longitude,
                    dropoff_latitude,
                    store_and_fwd_flag,
                    prediction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    payload["vendor_id"],
                    pickup_datetime,
                    payload["passenger_count"],
                    payload["pickup_longitude"],
                    payload["pickup_latitude"],
                    payload["dropoff_longitude"],
                    payload["dropoff_latitude"],
                    payload["store_and_fwd_flag"],
                    prediction,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)


def build_prediction_repository() -> PredictionRepository:
    repository = PredictionRepository(SETTINGS.db_path)
    repository.initialize()
    return repository
