from __future__ import annotations

import sqlite3

import pandas as pd

from ui.bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

import common

DB_PATH = common.CONFIG["paths"]["db_path"]
TARGET_COLUMN = common.CONFIG["dataset"]["target_column"]


def _load_duration_series(query: str, column_name: str) -> pd.Series:
    # All histogram pages use the same low-level SQLite reader so DB access
    # stays in one place and the Streamlit pages focus on presentation only.
    with sqlite3.connect(DB_PATH) as connection:
        data_frame = pd.read_sql(query, connection)

    if column_name not in data_frame.columns:
        return pd.Series(dtype="float64")

    return data_frame[column_name].dropna().astype(float)


def load_training_trip_durations() -> pd.Series:
    # Training distribution comes from the original train table target column.
    return _load_duration_series("SELECT trip_duration FROM train", TARGET_COLUMN)


def load_predicted_trip_durations() -> pd.Series:
    # Predicted distribution comes from persisted API inferences stored in SQLite.
    return _load_duration_series("SELECT prediction FROM predictions", "prediction")
