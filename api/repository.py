from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from api.config import SETTINGS
from api.model_metadata import ModelMetadata


class PredictionRepository:
    # Repository isolates SQLite concerns from the service and the routes.
    def __init__(self, db_path: str):
        self._db_path = db_path

    def initialize(self) -> None:
        with sqlite3.connect(self._db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS model_metadata (
                    version TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    model_created_at TEXT NOT NULL,
                    registered_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inference_timestamp TEXT NOT NULL,
                    vendor_id INTEGER NOT NULL,
                    pickup_datetime TEXT NOT NULL,
                    passenger_count INTEGER NOT NULL,
                    pickup_longitude REAL NOT NULL,
                    pickup_latitude REAL NOT NULL,
                    dropoff_longitude REAL NOT NULL,
                    dropoff_latitude REAL NOT NULL,
                    store_and_fwd_flag TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    model_version TEXT NOT NULL
                )
                """
            )
            self._ensure_prediction_columns(connection)
            connection.commit()

    def register_model_metadata(self, metadata: ModelMetadata) -> None:
        # Persist one row per model version so every prediction can point to it.
        registered_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as connection:
            connection.execute(
                """
                INSERT INTO model_metadata (
                    version,
                    model_path,
                    model_created_at,
                    registered_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(version) DO UPDATE SET
                    model_path = excluded.model_path,
                    model_created_at = excluded.model_created_at
                """,
                (
                    metadata.version,
                    metadata.path,
                    metadata.created_at,
                    registered_at,
                ),
            )
            connection.commit()

    def save_prediction(self, payload: dict, prediction: int, model_version: str) -> int:
        inference_timestamp = datetime.now(timezone.utc).isoformat()
        pickup_datetime = payload["pickup_datetime"]
        if hasattr(pickup_datetime, "isoformat"):
            pickup_datetime = pickup_datetime.isoformat()

        with sqlite3.connect(self._db_path) as connection:
            prediction_columns = self._get_prediction_columns(connection)
            insert_columns = []
            insert_values = []

            if "created_at" in prediction_columns:
                # Keep backward compatibility with legacy databases that still require created_at.
                insert_columns.append("created_at")
                insert_values.append(inference_timestamp)

            insert_columns.extend(
                [
                    "inference_timestamp",
                    "vendor_id",
                    "pickup_datetime",
                    "passenger_count",
                    "pickup_longitude",
                    "pickup_latitude",
                    "dropoff_longitude",
                    "dropoff_latitude",
                    "store_and_fwd_flag",
                    "prediction",
                    "model_version",
                ]
            )
            insert_values.extend(
                [
                    inference_timestamp,
                    payload["vendor_id"],
                    pickup_datetime,
                    payload["passenger_count"],
                    payload["pickup_longitude"],
                    payload["pickup_latitude"],
                    payload["dropoff_longitude"],
                    payload["dropoff_latitude"],
                    payload["store_and_fwd_flag"],
                    prediction,
                    model_version,
                ]
            )

            placeholders = ", ".join(["?"] * len(insert_columns))
            column_list = ", ".join(insert_columns)
            cursor = connection.execute(
                f"INSERT INTO predictions ({column_list}) VALUES ({placeholders})",
                tuple(insert_values),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def _ensure_prediction_columns(self, connection: sqlite3.Connection) -> None:
        # Support existing SQLite files without forcing a manual migration step.
        existing_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
        }

        if "inference_timestamp" not in existing_columns and "created_at" in existing_columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN inference_timestamp TEXT")
            connection.execute(
                "UPDATE predictions SET inference_timestamp = created_at "
                "WHERE inference_timestamp IS NULL"
            )

        if "model_version" not in existing_columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN model_version TEXT")

    def _get_prediction_columns(self, connection: sqlite3.Connection) -> set[str]:
        # Read the live schema because existing local databases may still be on the old layout.
        return {row[1] for row in connection.execute("PRAGMA table_info(predictions)").fetchall()}


def build_prediction_repository() -> PredictionRepository:
    repository = PredictionRepository(SETTINGS.db_path)
    repository.initialize()
    return repository
