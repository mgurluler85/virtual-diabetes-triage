from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_PAYLOAD = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "model_version": "v0.1",
    }


def test_predict_returns_numeric_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))


def test_predict_missing_feature_returns_422():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload.pop("s6")

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data
    assert any(
        error["loc"][-1] == "s6"
        for error in data["detail"]
    )