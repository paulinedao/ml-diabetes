import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock



VALID_PATIENT = {
    "Pregnancies": 2,
    "Glucose": 138.0,
    "BloodPressure": 62.0,
    "SkinThickness": 35.0,
    "Insulin": 0.0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 47.0,
}


def make_mock_artifacts():
    """Return mock imputer, scaler and model that behave like the real ones."""
    imputer = MagicMock()
    imputer.transform.side_effect = lambda x: x  # pass-through

    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 8))

    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.28, 0.72]])

    return imputer, scaler, model


# Model artifact tests 

class TestModelArtifacts:
    """Make sure the saved .pkl files load and behave as expected."""

    def test_artifacts_load(self):
        import joblib, os
        assert os.path.exists("predict/imputer.pkl"), "imputer.pkl not found"
        assert os.path.exists("predict/scaler.pkl"),  "scaler.pkl not found"
        assert os.path.exists("predict/model.pkl"),   "model.pkl not found"

        imputer = joblib.load("predict/imputer.pkl")
        scaler  = joblib.load("predict/scaler.pkl")
        model   = joblib.load("predict/model.pkl")

        assert imputer is not None
        assert scaler  is not None
        assert model   is not None

    def test_model_predicts_shape(self):
        import joblib
        import pandas as pd

        imputer = joblib.load("predict/imputer.pkl")
        scaler  = joblib.load("predict/scaler.pkl")
        model   = joblib.load("predict/model.pkl")

        COLUMNS_TO_IMPUTE = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        ALL_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

        data = pd.DataFrame([VALID_PATIENT])
        data[COLUMNS_TO_IMPUTE] = data[COLUMNS_TO_IMPUTE].replace(0, np.nan)
        data[COLUMNS_TO_IMPUTE] = imputer.transform(data[COLUMNS_TO_IMPUTE])
        scaled = scaler.transform(data[ALL_FEATURES])

        prediction = model.predict(scaled)
        probability = model.predict_proba(scaled)

        assert prediction.shape == (1,)
        assert probability.shape == (1, 2)
        assert prediction[0] in (0, 1)
        assert 0.0 <= probability[0][1] <= 1.0


# API endpoint tests (mocked artifacts)

class TestPredictEndpoint:
    """Test the /predict endpoint with mocked ML artifacts."""

    @pytest.fixture(autouse=True)
    def client(self):
        imputer, scaler, model = make_mock_artifacts()
        with patch("app.imputer", imputer), \
             patch("app.scaler",  scaler),  \
             patch("app.model",   model):
            from app import app
            self.client = TestClient(app)

    def test_predict_returns_200(self):
        res = self.client.post("/predict", json=VALID_PATIENT)
        assert res.status_code == 200

    def test_predict_response_shape(self):
        res = self.client.post("/predict", json=VALID_PATIENT)
        body = res.json()
        assert "diabetic" in body
        assert "probability" in body

    def test_predict_diabetic_is_bool(self):
        res = self.client.post("/predict", json=VALID_PATIENT)
        assert isinstance(res.json()["diabetic"], bool)

    def test_predict_probability_in_range(self):
        res = self.client.post("/predict", json=VALID_PATIENT)
        prob = res.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_missing_field_returns_422(self):
        incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "Glucose"}
        res = self.client.post("/predict", json=incomplete)
        assert res.status_code == 422

    def test_invalid_field_type_returns_422(self):
        bad = {**VALID_PATIENT, "Age": "not-a-number"}
        res = self.client.post("/predict", json=bad)
        assert res.status_code == 422

    def test_get_predict_returns_405(self):
        res = self.client.get("/predict")
        assert res.status_code == 405

    def test_zero_values_accepted(self):
        zero_patient = {k: 0 for k in VALID_PATIENT}
        res = self.client.post("/predict", json=zero_patient)
        assert res.status_code == 200