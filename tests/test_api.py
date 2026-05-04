from fastapi.testclient import TestClient

from api import load_model_and_metadata
from main import app


client = TestClient(app)


def setup_module(module):
    # Ensure model artifacts are loaded for direct endpoint tests.
    assert load_model_and_metadata()


def test_health_endpoint_returns_expected_shape():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()

    assert body["status"] in {"healthy", "unhealthy"}
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["metadata_loaded"], bool)
    assert isinstance(body["message"], str)


def test_predict_endpoint_returns_prediction_payload():
    payload = {
        "total_images": 10,
        "beds": 3,
        "baths": 2.5,
        "area": 1800.0,
        "latitude": 40.7128,
        "longitude": -74.006,
        "garden": 1,
        "garage": 1,
        "new_construction": 0,
        "pool": 0,
        "terrace": 1,
        "air_conditioning": 1,
        "parking": 1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert isinstance(body["predicted_price"], float)
    assert body["currency"] == "USD"
    assert body["model_version"] == "1.0.0"
