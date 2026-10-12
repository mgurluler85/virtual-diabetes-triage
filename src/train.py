import json
from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def train():
    # Load the diabetes regression dataset
    diabetes = load_diabetes(as_frame=True)

    X = diabetes.frame.drop(columns=["target"])
    y = diabetes.frame["target"]

    # Create a deterministic train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # v0.2 improved model
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on the held-out test set
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)

    # Create artifact directory if it does not exist
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    # Save evaluation metrics
    metrics = {
        "model_version": "v0.2",
        "model": "StandardScaler + Ridge(alpha=1.0)",
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "rmse": float(rmse),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Test RMSE: {rmse:.4f}")


if __name__ == "__main__":
    train()