# Virtual Diabetes Clinic Triage

A small MLOps project that trains and deploys a machine learning model for short-term diabetes progression prediction.

The project uses the scikit-learn Diabetes regression dataset as a stand-in for de-identified EHR features. It is intended for educational purposes only and is not a clinical decision-support system.

## Model

### v0.1 — Baseline

- StandardScaler
- LinearRegression
- 80/20 train-test split
- random_state=42
- Held-out test RMSE: `53.8534`

### v0.2 — Improved Model

- StandardScaler
- Ridge(alpha=1.0)
- Same 80/20 train-test split
- Same random_state=42
- Held-out test RMSE: `53.7775`

The RMSE improved from `53.8534` in v0.1 to `53.7775` in v0.2, a delta of `-0.0759` (lower is better).

RandomForestRegressor was also evaluated and produced an RMSE of `54.4612`, so Ridge was selected for v0.2.

## Input Features

The model expects the 10 features from the scikit-learn Diabetes dataset:

```text
age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
```

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train.py
```

Training creates:

```text
artifacts/model.joblib
artifacts/metrics.json
```

## Run Tests

```bash
python -m pytest -v
```

Run linting:

```bash
python -m ruff check .
```

## Run the API Locally

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "model_version": "v0.2"
}
```

## Prediction

Send a POST request to `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001
  }'
```

Example response:

```json
{
  "prediction": 235.94963722176274
}
```

## Validation Error

All 10 features are required.

For example, if `s6` is missing, the API returns HTTP 422 with a JSON validation error containing a structure similar to:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "s6"],
      "msg": "Field required"
    }
  ]
}
```

## Docker

Build the Docker image locally:

```bash
docker build -t virtual-diabetes-triage:v0.2 .
```

Run the container:

```bash
docker run --rm -p 80:80 virtual-diabetes-triage:v0.2
```

Test it:

```bash
curl http://localhost:80/health
```

## Published Docker Image

The v0.2 image is publicly available from GitHub Container Registry:

```bash
docker pull ghcr.io/mgurluler85/virtual-diabetes-triage:v0.2
```

Run the published image:

```bash
docker run --rm -p 80:80 ghcr.io/mgurluler85/virtual-diabetes-triage:v0.2
```

Then test:

```bash
curl http://localhost:80/health
```

## CI/CD

GitHub Actions is used for continuous integration and release automation.

On pushes and pull requests to `main`, CI:

- installs the pinned dependencies;
- runs Ruff linting;
- runs the unit tests;
- performs a training smoke test;
- uploads the model and metrics as workflow artifacts.

When a Git tag matching `v*` is pushed, the release workflow:

- builds the Docker image;
- starts the container;
- smoke-tests `/health`;
- smoke-tests `/predict`;
- verifies invalid input returns HTTP 422;
- pushes the versioned image to GitHub Container Registry;
- creates a GitHub Release.

## Reproducibility

The project uses:

- pinned Python dependencies in `requirements.txt`;
- a fixed `random_state=42`;
- a fixed 80/20 train-test split;
- saved model metrics in `artifacts/metrics.json`;
- versioned Docker images.

## Disclaimer

The scikit-learn Diabetes dataset is used only as a stand-in for de-identified EHR data. This project is for educational purposes and must not be used for real clinical decision-making.
