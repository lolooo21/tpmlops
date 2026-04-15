from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from model.preprocessing.preprocessing import add_model_features, transform_target
from model.preprocessing.ridge_features import TRAIN_FEATURES


class TaxiTripDurationModel:
    # This wrapper keeps inference concerns inside the model artifact,
    # so the API only has to call predict on one object.
    def __init__(self, model: Pipeline):
        self.model = model
        self.abnormal_dates: list[str] = []

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        # Training and serving both go through the same feature engineering path.
        return add_model_features(X, abnormal_dates=self.abnormal_dates)

    def preprocess_target(self, y: pd.Series) -> pd.Series:
        return transform_target(y)

    def postprocess(self, raw_predictions: np.ndarray) -> np.ndarray:
        # The regression is trained in log space and returned in seconds.
        predictions = np.expm1(raw_predictions)
        return np.maximum(np.round(predictions), 0).astype(int)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TaxiTripDurationModel:
        raise NotImplementedError("Use the training script to fit this model.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_processed = self.preprocess(X)
        raw_predictions = self.model.predict(X_processed[TRAIN_FEATURES])
        return self.postprocess(raw_predictions)
