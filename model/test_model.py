import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.load_data import load_random_test_data
import common
from model.custom_model import TaxiTripDurationModel

MODEL_PATH = common.CONFIG["paths"]["model_custom_path"]
TARGET_COLUMN = common.CONFIG["dataset"]["target_column"]


def load_model(path):
    # Helper kept separate so local tests can point to another artifact if needed.
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model = pickle.load(file)
    if not isinstance(model, TaxiTripDurationModel):
        raise TypeError("Expected a TaxiTripDurationModel artifact.")
    print("Done")
    return model


def test_model(model):
    # Quick manual smoke test against a few rows sampled from the test set.
    print("Testing the model")
    X, y = load_random_test_data()
    y_pred = model.predict(X)
    df = X.copy()
    df[TARGET_COLUMN] = y
    df["prediction"] = y_pred
    print(df)


if __name__ == "__main__":
    # Convenience entry point for local validation after retraining.
    model = load_model(MODEL_PATH)
    test_model(model)
