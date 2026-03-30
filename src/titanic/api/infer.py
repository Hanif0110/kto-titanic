"""
Ce script permet d'inférer le model de machine learning et de le mettre à disposition
dans un Webservice. Il pourra donc être utilisé par notre chatbot par exemple,
ou directement par un front. Remplir ce script une fois l'entrainement du model fonctionne
"""

import os
from dataclasses import dataclass
from enum import Enum

import joblib
import pandas as pd
from fastapi import Depends, FastAPI

from titanic.api.auth import verify_token


JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://jaeger.hanif0110-dev.svc.cluster.local:4318/v1/traces")

app = FastAPI()

model = joblib.load("./src/titanic/api/resources/model.pkl")


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
    df_passenger = pd.DataFrame([passenger.to_dict()])

    df_passenger["Sex"] = pd.Categorical(
        df_passenger["Sex"],
        categories=[Sex.FEMALE.value, Sex.MALE.value],
    )

    df_to_predict = pd.get_dummies(df_passenger)

    if hasattr(model, "feature_names_in_"):
        expected_columns = list(model.feature_names_in_)
        df_to_predict = df_to_predict.reindex(columns=expected_columns, fill_value=0)

    res = model.predict(df_to_predict)

    return res.tolist()