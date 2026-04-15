from __future__ import annotations

import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common
from api.model_metadata import build_model_metadata, save_model_metadata
from data.load_data import load_test_data, load_train_data
from model.inference.custom_model import TaxiTripDurationModel
from model.preprocessing.features import build_abnormal_dates
from model.preprocessing.preprocessing import transform_target
from model.preprocessing.ridge_features import CAT_FEATURES, NUM_FEATURES, TRAIN_FEATURES

MODEL_PATH = common.CONFIG["paths"]["model_custom_path"]
MODEL_METADATA_PATH = common.CONFIG["paths"]["model_custom_metadata_path"]
MODEL_VERSIONS_DIR = common.CONFIG["paths"]["model_custom_versions_dir"]


def build_model() -> Pipeline:
    # The sklearn pipeline focuses on model-native transforms only.
    column_transformer = ColumnTransformer(
        [
            ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("scaling", StandardScaler(), NUM_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("ohe_and_scaling", column_transformer),
            ("regression", Ridge()),
        ]
    )


def train_model() -> TaxiTripDurationModel:
    print("Building custom model")

    X_train, y_train = load_train_data()
    X_test, y_test = load_test_data()

    model = TaxiTripDurationModel(build_model())
    abnormal_dates = build_abnormal_dates(X_train)
    model.abnormal_dates = [str(date) for date in abnormal_dates]

    X_train_processed = model.preprocess(X_train)
    y_train_processed = model.preprocess_target(y_train)
    model.model.fit(X_train_processed[TRAIN_FEATURES], y_train_processed)

    y_train_log = transform_target(y_train)
    y_test_log = transform_target(y_test)
    X_test_processed = model.preprocess(X_test)

    y_pred_train = model.model.predict(X_train_processed[TRAIN_FEATURES])
    y_pred_test = model.model.predict(X_test_processed[TRAIN_FEATURES])

    print(f"Train RMSLE = {root_mean_squared_error(y_train_log, y_pred_train):.4f}")
    print(f"Test RMSLE = {root_mean_squared_error(y_test_log, y_pred_test):.4f}")
    print(f"Test R2 = {r2_score(y_test_log, y_pred_test):.4f}")

    return model


def persist_model(model: TaxiTripDurationModel, path: str) -> None:
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print("Done")


def build_model_version() -> str:
    # Timestamp-based versioning keeps each training run traceable without manual edits.
    trained_at = datetime.now(timezone.utc)
    return trained_at.strftime("taxi_trip_duration_custom_%Y%m%dT%H%M%SZ")


def persist_training_artifacts(model: TaxiTripDurationModel, model_path: str, metadata_path: str) -> None:
    # Store one immutable versioned artifact and refresh the latest aliases used by the API.
    model_version = build_model_version()
    versions_dir = Path(MODEL_VERSIONS_DIR)
    versions_dir.mkdir(parents=True, exist_ok=True)

    versioned_model_path = versions_dir / f"{model_version}.model"
    versioned_metadata_path = versions_dir / f"{model_version}.metadata.json"

    persist_model(model, str(versioned_model_path))
    versioned_metadata = build_model_metadata(str(versioned_model_path), model_version)
    save_model_metadata(versioned_metadata, str(versioned_metadata_path))

    shutil.copy2(versioned_model_path, model_path)
    latest_metadata = build_model_metadata(model_path, model_version)
    save_model_metadata(latest_metadata, metadata_path)
    print(f"Model version = {model_version}")


if __name__ == "__main__":
    # Script entry point used by the API-ready custom wrapper.
    model = train_model()
    persist_training_artifacts(model, MODEL_PATH, MODEL_METADATA_PATH)
