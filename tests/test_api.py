import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_predict():

    response = client.post(
        "/predict",
        json={
            "Sales":200,
            "Discount":0.1,
            "Year":2026,
            "Month":5,
            "WeekDay":2
        }
    )

    assert response.status_code == 200

    data=response.json()

    assert "predicted_profit" in data