"""
Ce script permet d'inférer le modèle de machine learning et de le mettre à disposition
sous forme de Webservice FastAPI.
"""

import os
import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status

from titanic.api.auth import verify_token


JAEGER_ENDPOINT = os.getenv(
    "JAEGER_ENDPOINT",
    "http://jaeger.hanif0110-dev.svc.cluster.local:4318/v1/traces",
)

MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_RESOURCE_DIR = Path("./src/titanic/api/resources")

app = FastAPI()

model: Any | None = None


def find_model_path() -> Path:
    """Trouve le modèle téléchargé depuis MLflow."""
    if MODEL_PATH:
        return Path(MODEL_PATH)

    candidates = [
        *MODEL_RESOURCE_DIR.rglob("model.pkl"),
        *MODEL_RESOURCE_DIR.rglob("*.joblib"),
        *MODEL_RESOURCE_DIR.rglob("*.pkl"),
    ]

    if candidates:
        return candidates[0]

    return MODEL_RESOURCE_DIR / "model.pkl"


def load_model() -> Any:
    """Charge le modèle depuis les ressources locales."""
    model_path = find_model_path()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.suffix == ".joblib":
        return joblib.load(model_path)

    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


def get_model() -> Any:
    """
    Charge le modèle seulement au moment de l'inférence.

    Cela évite de faire échouer les tests unitaires GitHub Actions,
    car le fichier model.pkl n'existe pas encore au début de la pipeline.
    """
    global model

    if model is not None:
        return model

    try:
        model = load_model()
        return model
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available yet: {exc}",
        ) from exc


class Pclass(Enum):
    UPPER = 1
    MIDDLE = 2
    LOW = 3


class Sex(Enum):
    MALE = "male"
    FEMALE = "female"


@dataclass
class Passenger:
    pclass: Pclass
    sex: Sex
    sibSp: int
    parch: int

    def to_dict(self) -> dict:
        return {
            "Pclass": self.pclass.value,
            "Sex": self.sex.value,
            "SibSp": self.sibSp,
            "Parch": self.parch,
        }


@app.get("/health")
def health() -> dict:
    return {"status": "OK"}


@app.post("/infer")
def infer(passenger: Passenger, token: str = Depends(verify_token("api:read"))) -> list:
    loaded_model = get_model()

    df_passenger = pd.DataFrame([passenger.to_dict()])

    df_passenger["Sex"] = pd.Categorical(
        df_passenger["Sex"],
        categories=[Sex.FEMALE.value, Sex.MALE.value],
    )

    df_to_predict = pd.get_dummies(df_passenger)

    if hasattr(loaded_model, "feature_names_in_"):
        expected_columns = list(loaded_model.feature_names_in_)
        df_to_predict = df_to_predict.reindex(columns=expected_columns, fill_value=0)

    res = loaded_model.predict(df_to_predict)

    return res.tolist()