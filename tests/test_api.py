from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


TEST_APPLICATION = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Employment_Status": "Salaried",
    "Applicant_Income": 50000,
    "Coapplicant_Income": 20000,
    "Loan_Amount": 150000,
    "Loan_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban",
    "Age": 30
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_predict():
    response = client.post(
        "/predict",
        json=TEST_APPLICATION
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "approval_probability" in data

    assert data["prediction"] in [
        "Approved",
        "Rejected"
    ]

    assert 0 <= data["approval_probability"] <= 1


def test_explain():
    response = client.post(
        "/explain",
        json=TEST_APPLICATION
    )

    assert response.status_code == 200

    data = response.json()

    assert "explanations" in data
    assert isinstance(data["explanations"], list)

    if data["explanations"]:
        explanation = data["explanations"][0]

        assert "feature" in explanation
        assert "value" in explanation
        assert "contribution" in explanation
        assert "direction" in explanation


def test_counterfactual():
    response = client.post(
        "/counterfactual",
        json=TEST_APPLICATION
    )

    assert response.status_code == 200

    data = response.json()

    assert "original_prediction" in data
    assert "original_probability" in data
    assert "counterfactuals" in data
    assert "found" in data

    assert data["original_prediction"] in [
        "Approved",
        "Rejected"
    ]

    assert 0 <= data["original_probability"] <= 1
    assert isinstance(data["counterfactuals"], list)
    assert isinstance(data["found"], bool)


def test_analyze():
    response = client.post(
        "/analyze",
        json=TEST_APPLICATION
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "approval_probability" in data
    assert "explanations" in data
    assert "original_prediction" in data
    assert "original_probability" in data
    assert "counterfactuals" in data
    assert "counterfactual_found" in data

    assert data["prediction"] in [
        "Approved",
        "Rejected"
    ]

    assert 0 <= data["approval_probability"] <= 1
    assert 0 <= data["original_probability"] <= 1

    assert isinstance(data["explanations"], list)
    assert isinstance(data["counterfactuals"], list)
    assert isinstance(data["counterfactual_found"], bool)