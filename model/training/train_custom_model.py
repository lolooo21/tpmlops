from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_data import load_test_data, load_train_data
from model.inference.custom_model import TaxiTripDurationModel
from model.preprocessing.features import build_abnormal_dates
from model.preprocessing.preprocessing import transform_target
from model.preprocessing.ridge_features import CAT_FEATURES, NUM_FEATURES, TRAIN_FEATURES

import common

MODEL_PATH = common.CONFIG["paths"]["model_custom_path"]


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


if __name__ == "__main__":
    # Script entry point used by the API-ready custom wrapper.
    model = train_model()
    persist_model(model, MODEL_PATH)
