import pandas as pd

import json

import joblib

from src.train import METRICS_PATH, MODEL_PATH, train


def test_training_creates_artifacts():
    train()

    assert MODEL_PATH.exists()
    assert METRICS_PATH.exists()


def test_saved_model_can_make_prediction():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        [[
            0.02,
            -0.044,
            0.06,
            -0.03,
            -0.02,
            0.03,
            -0.02,
            0.02,
            0.02,
            -0.001,
        ]],
        columns=[
            "age",
            "sex",
            "bmi",
            "bp",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "s6",
        ],
    )

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert isinstance(float(prediction[0]), float)


def test_metrics_are_saved_correctly():
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["model_version"] == "v0.1"
    assert metrics["model"] == "StandardScaler + LinearRegression"
    assert metrics["random_state"] == 42
    assert metrics["test_size"] == 0.2
    assert isinstance(metrics["rmse"], float)
    assert metrics["rmse"] > 0