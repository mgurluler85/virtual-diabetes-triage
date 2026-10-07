from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path("artifacts/model.joblib")
MODEL_VERSION = "v0.1"

FEATURE_NAMES = [
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
]


model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Virtual Diabetes Clinic Triage API",
    version=MODEL_VERSION,
)


class PatientFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def predict(patient: PatientFeatures):
    input_data = pd.DataFrame(
        [[getattr(patient, feature) for feature in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    prediction = model.predict(input_data)[0]

    return {
        "prediction": float(prediction),
    }