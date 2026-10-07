
# Changelog

All notable changes to this project are documented in this file.

## v0.2

### Model
- Replaced the v0.1 `StandardScaler + LinearRegression` baseline with `StandardScaler + Ridge(alpha=1.0)`.
- Used the same 80/20 train-test split and `random_state=42` for a fair comparison.
- v0.1 RMSE: `53.8534`.
- v0.2 RMSE: `53.7775`.
- RMSE delta: `-0.0759` (lower is better).
- RandomForestRegressor was also evaluated but produced a higher RMSE (`54.4612`), so Ridge was selected for v0.2.

### API
- Updated the reported model version from `v0.1` to `v0.2`.

### Validation
- All 6 unit tests pass.
- Ruff linting passes.

## v0.1

### Model
- Added the baseline diabetes progression prediction model.
- Uses `StandardScaler` followed by `LinearRegression`.
- Uses an 80/20 train-test split with `random_state=42`.
- Held-out test RMSE: `53.8534`.

### API
- Added a FastAPI service.
- Added `GET /health`.
- Added `POST /predict`.
- Added input validation for all 10 diabetes dataset features.
- Missing required features return HTTP 422.

### Testing
- Added API unit tests.
- Added training and artifact tests.
- Added linting with Ruff.

### MLOps
- Added reproducible model training.
- Added GitHub Actions CI for linting, tests, and training smoke tests.
- Added training artifact upload in CI.
- Added a Docker image containing the trained model.
- Added a tag-triggered release workflow.
- Added container smoke tests before publishing.
- Published the v0.1 Docker image to GitHub Container Registry.
