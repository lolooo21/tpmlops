from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_data import load_train_data, load_test_data
from model.features import build_abnormal_dates
from model.preprocessing import add_model_features, transform_target
from model.ridge_features import CAT_FEATURES, NUM_FEATURES, TRAIN_FEATURES
from model.serving import TaxiTripDurationModel

import common

MODEL_PATH = common.CONFIG["paths"]["model_ridge_path"]


def build_model():
    # The sklearn pipeline handles only transformations that belong to the model itself:
    # categorical encoding, numeric scaling and regression.
    column_transformer = ColumnTransformer(
        [
            ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("scaling", StandardScaler(), NUM_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("ohe_and_scaling", column_transformer),
            ("regression", Ridge()),
        ]
    )
    return pipeline


def train_model():
    print("Building ridge model")

    # The train/test split already exists in SQLite and is reused here as-is.
    X_train, y_train = load_train_data()
    X_test, y_test = load_test_data()

    # This state must be persisted because it is needed later by the API.
    abnormal_dates = build_abnormal_dates(X_train)

    X_train = add_model_features(X_train, abnormal_dates)
    X_test = add_model_features(X_test, abnormal_dates)

    # The model optimizes a log-transformed target to stabilize long trip durations.
    y_train = transform_target(y_train)
    y_test = transform_target(y_test)

    pipeline = build_model()
    pipeline.fit(X_train[TRAIN_FEATURES], y_train)

    y_pred_train = pipeline.predict(X_train[TRAIN_FEATURES])
    y_pred_test = pipeline.predict(X_test[TRAIN_FEATURES])

    print(f"Train RMSLE = {root_mean_squared_error(y_train, y_pred_train):.4f}")
    print(f"Test RMSLE = {root_mean_squared_error(y_test, y_pred_test):.4f}")
    print(f"Test R2 = {r2_score(y_test, y_pred_test):.4f}")

    # Persist a serving artifact instead of the raw pipeline so the API does not
    # need to know how to rebuild preprocessing state.
    return TaxiTripDurationModel(
        pipeline=pipeline,
        abnormal_dates=[str(date) for date in abnormal_dates],
    )


def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print("Done")


if __name__ == "__main__":
    # Script entry point used to regenerate the API-ready model artifact.
    model = train_model()
    persist_model(model, MODEL_PATH)
