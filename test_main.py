from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metadata_check():
    response = client.get("/metadata")
    assert response.status_code == 200
    assert "last_trained_timestamp" in response.json()


def test_predict_success():
    payload = {
        "social_ad_spend": 5000.0,
        "search_ad_spend": 3000.0,
        "discount_percent": 15.0,
        "is_holiday": 0,
        "industry": "Tech",
        "location": "Bengaluru"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_sales" in response.json()
    assert isinstance(response.json()["predicted_sales"], float)


def test_predict_invalid_enum():
    payload = {
        "social_ad_spend": 5000.0,
        "search_ad_spend": 3000.0,
        "discount_percent": 15.0,
        "is_holiday": 0,
        "industry": "Technology",
        "location": "Bengaluru"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
