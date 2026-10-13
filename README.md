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

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python src/train.py
```

This trains the v0.2 Ridge model and creates:

```text
artifacts/model.joblib
artifacts/metrics.json
```

The expected held-out test RMSE is approximately:

```text
53.7775
```

## Run Tests

Run the unit tests:

```bash
python -m pytest -v
```

Run linting:

```bash
python -m ruff check .
```

## Run API Locally

Start the FastAPI service:

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

Send all 10 required features to `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.0380759064,
    "sex": 0.0506801187,
    "bmi": 0.0616962065,
    "bp": 0.0218723855,
    "s1": -0.0442234984,
    "s2": -0.0348207628,
    "s3": -0.0434008457,
    "s4": -0.0025922620,
    "s5": 0.0199074862,
    "s6": -0.0176461252
  }'
```

Example response:

```json
{
  "prediction": 226.91314681729324
}
```

The exact prediction is a floating-point value returned by the currently loaded model.

## Validation Error

All 10 input features are required.

For example, if `s6` is omitted:

```bash
curl -i -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.0380759064,
    "sex": 0.0506801187,
    "bmi": 0.0616962065,
    "bp": 0.0218723855,
    "s1": -0.0442234984,
    "s2": -0.0348207628,
    "s3": -0.0434008457,
    "s4": -0.0025922620,
    "s5": 0.0199074862
  }'
```

The API returns HTTP `422 Unprocessable Entity` with a JSON validation error identifying the missing field, for example:

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

Build the current version locally:

```bash
docker build -t virtual-diabetes-triage:v0.2 .
```

Run it:

```bash
docker run --rm -p 80:80 virtual-diabetes-triage:v0.2
```

Test it:

```bash
curl http://localhost:80/health
```

## Published Docker Images

Both released Docker images are publicly available from GitHub Container Registry.

### v0.1

Pull the image:

```bash
docker pull ghcr.io/mgurluler85/virtual-diabetes-triage:v0.1
```

Run the container:

```bash
docker run -it -p 80:80 ghcr.io/mgurluler85/virtual-diabetes-triage:v0.1
```

Health check:

```bash
curl http://localhost:80/health
```

Expected response:

```json
{
  "status": "ok",
  "model_version": "v0.1"
}
```

Stop the v0.1 container before starting v0.2 because both versions use port 80.

### v0.2

Pull the image:

```bash
docker pull ghcr.io/mgurluler85/virtual-diabetes-triage:v0.2
```

Run the container:

```bash
docker run -it -p 80:80 ghcr.io/mgurluler85/virtual-diabetes-triage:v0.2
```

Health check:

```bash
curl http://localhost:80/health
```

Expected response:

```json
{
  "status": "ok",
  "model_version": "v0.2"
}
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
- verifies that a request with missing `s6` returns HTTP 422;
- pushes the versioned Docker image to GitHub Container Registry;
- creates a GitHub Release.

## Reproducibility

The project uses:

- pinned dependency versions in `requirements.txt`;
- `random_state=42` for the train-test split;
- the same fixed 80/20 split for model comparison;
- versioned model metrics;
- versioned Docker images;
- automated tests and training in GitHub Actions.

The v0.2 model can be reproduced locally with:

```bash
python src/train.py
```

The resulting metrics are saved to:

```text
artifacts/metrics.json
```

## Version History

### v0.1

Baseline release using StandardScaler and LinearRegression.

Held-out test RMSE:

```text
53.8534
```

### v0.2

Improved release using StandardScaler and Ridge regression with `alpha=1.0`.

Held-out test RMSE:

```text
53.7775
```

RandomForestRegressor was also evaluated with an RMSE of `54.4612`. Ridge was selected because it achieved the lowest held-out RMSE among the evaluated models.

See `CHANGELOG.md` for detailed version changes.

## Disclaimer

This project is for educational purposes only and must not be used for clinical decision-making.