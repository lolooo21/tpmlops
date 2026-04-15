from __future__ import annotations

import pickle
import sqlite3
from functools import lru_cache

import pandas as pd

from api.config import SETTINGS
from api.model_metadata import ModelMetadata, load_model_metadata
from api.repository import build_prediction_repository
from model.inference.custom_model import TaxiTripDurationModel


class PredictionService:
    # Service layer isolates I/O and inference from the FastAPI routes.
    def __init__(self, model: TaxiTripDurationModel, repository, model_metadata: ModelMetadata):
        self._model = model
        self._repository = repository
        self._model_metadata = model_metadata

    def predict(self, payload: dict) -> tuple[int, int]:
        # The model artifact expects a dataframe to reuse training-time feature logic.
        input_frame = pd.DataFrame([payload])
        prediction = self._model.predict(input_frame)[0]
        prediction = int(prediction)
        prediction_id = self._repository.save_prediction(
            payload,
            prediction,
            self._model_metadata.version,
        )
        return prediction_id, prediction

    def get_random_test_row(self) -> tuple[dict, float]:
        # This endpoint is useful to inspect a realistic payload quickly.
        with sqlite3.connect(SETTINGS.db_path) as connection:
            data_test = pd.read_sql("SELECT * FROM test ORDER BY RANDOM() LIMIT 1", connection)

        y = float(data_test[SETTINGS.target_column].iloc[0])
        x = data_test.drop(columns=[SETTINGS.target_column]).iloc[0].to_dict()
        return x, y


@lru_cache
def load_prediction_service() -> PredictionService:
    # Cache the service so the model is loaded once per process.
    with open(SETTINGS.model_path, "rb") as file:
        model = pickle.load(file)

    if not isinstance(model, TaxiTripDurationModel):
        # This protects the API from loading an incompatible artifact by mistake.
        raise TypeError(
            "The persisted model must be a TaxiTripDurationModel. "
            "Train it again with `python -m model.training.train_custom_model`."
        )

    model_metadata = load_model_metadata(SETTINGS.model_metadata_path)
    repository = build_prediction_repository()
    repository.register_model_metadata(model_metadata)
    return PredictionService(model, repository, model_metadata)
