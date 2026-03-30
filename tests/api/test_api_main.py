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
    from titanic.api.infer import app, Pclass, Sex, Passenger
    from titanic.api.main import main
    from titanic.api import infer


@pytest.fixture
def client():
    return TestClient(app)


def test_api_main_is_runnable():
    with patch("uvicorn.run") as mock_run:
        main()
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["host"] == "0.0.0.0"
        assert call_args[1]["port"] == 8080


def test_api_infer_module_is_importable():
    assert hasattr(infer, "app")
    assert hasattr(infer, "infer")
    assert hasattr(infer, "health")


def test_api_has_required_enums():
    assert hasattr(Pclass, "UPPER")
    assert hasattr(Pclass, "MIDDLE")
    assert hasattr(Pclass, "LOW")

    assert Pclass.UPPER.value == 1
    assert Pclass.MIDDLE.value == 2
    assert Pclass.LOW.value == 3

    assert Sex.MALE.value == "male"
    assert Sex.FEMALE.value == "female"


def test_api_passenger_dataclass():
    passenger = Passenger(pclass=Pclass.UPPER, sex=Sex.FEMALE, sibSp=1, parch=2)
    passenger_dict = passenger.to_dict()

    assert passenger_dict["Pclass"] == 1
    assert passenger_dict["Sex"] == "female"
    assert passenger_dict["SibSp"] == 1
    assert passenger_dict["Parch"] == 2


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}