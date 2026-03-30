from unittest.mock import Mock, patch
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient


mock_model = Mock()
mock_model.predict.return_value = np.array([1])
mock_model.feature_names_in_ = np.array(["Pclass", "SibSp", "Parch", "Sex_female", "Sex_male"])


with (
    patch("joblib.load", return_value=mock_model),
    patch.dict(os.environ, {"OAUTH2_DOMAIN": ""}, clear=False),
):
    from titanic.api.infer import app


@pytest.fixture(autouse=True)
def reset_oauth_env():
    with patch.dict(os.environ, {"OAUTH2_DOMAIN": ""}, clear=False):
        yield


@pytest.fixture
def mock_infer_model():
    model = Mock()
    model.predict.return_value = np.array([1])
    model.feature_names_in_ = np.array(["Pclass", "SibSp", "Parch", "Sex_female", "Sex_male"])

    with patch("titanic.api.infer.model", model):
        yield model


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_infer_first_class_female(client, mock_infer_model):
    mock_infer_model.predict.return_value = np.array([1])
    payload = {"pclass": 1, "sex": "female", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert response.json() == [1]
    mock_infer_model.predict.assert_called_once()


def test_infer_third_class_male(client, mock_infer_model):
    mock_infer_model.reset_mock()
    mock_infer_model.predict.return_value = np.array([0])
    payload = {"pclass": 3, "sex": "male", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert response.json() == [0]
    mock_infer_model.predict.assert_called_once()


def test_infer_with_family(client, mock_infer_model):
    mock_infer_model.reset_mock()
    mock_infer_model.predict.return_value = np.array([1])
    payload = {"pclass": 2, "sex": "female", "sibSp": 1, "parch": 2}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert response.json() == [1]
    mock_infer_model.predict.assert_called_once()


def test_infer_invalid_pclass(client):
    payload = {"pclass": 5, "sex": "female", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 422


def test_infer_invalid_sex(client):
    payload = {"pclass": 1, "sex": "unknown", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 422


def test_infer_missing_field(client):
    payload = {"pclass": 1, "sex": "male", "sibSp": 0}
    response = client.post("/infer", json=payload, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 422


def test_infer_without_token(client):
    payload = {"pclass": 1, "sex": "female", "sibSp": 0, "parch": 0}
    response = client.post("/infer", json=payload)
    assert response.status_code == 401