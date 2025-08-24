# challenge/api.py
import pandas as pd
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST
from pydantic import BaseModel, validator

from challenge.model import DelayModel


# -----------------------------------------------------------------------------
# App & global model
# -----------------------------------------------------------------------------
app = FastAPI()
model = DelayModel()

# Ruta robusta al dataset (…/challenge_MLE/data/data.csv)
_DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "data.csv"


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def _serialize_validation_errors(errors):
    """
    Convierte los errores de validación a un dict JSON-serializable
    (evita incluir objetos Exception crudos).
    """
    sanitized = []
    for err in errors:
        e = dict(err)
        if "ctx" in e and e["ctx"]:
            e["ctx"] = {
                k: (str(v) if isinstance(v, Exception) else v)
                for k, v in e["ctx"].items()
            }
        sanitized.append(e)
    return sanitized


def _ensure_trained():
    """
    Entrena el modelo si aún no está entrenado.
    Se usa en startup y como guard en /predict (por si startup no corre en tests).
    """
    if model._model is None:
        df = pd.read_csv(_DATA_FILE)
        X, y = model.preprocess(df, target_column="delay")
        model.fit(X, y)


# -----------------------------------------------------------------------------
# Handlers de error
# -----------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    # Convierte 422 -> 400 y sanitiza el payload
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": _serialize_validation_errors(exc.errors())},
    )


# -----------------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    try:
        _ensure_trained()
        print("Model trained successfully (startup)!")
    except Exception as e:
        # No aborta el arranque; /predict hará lazy-train si hace falta
        print(f"Startup training failed (will lazy-train on demand): {e}")


# -----------------------------------------------------------------------------
# Esquemas de entrada (Pydantic v1)
# -----------------------------------------------------------------------------
class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("MES")
    def validate_month(cls, v):
        if not 1 <= v <= 12:
            raise ValueError("MES must be between 1 and 12")
        return v

    @validator("TIPOVUELO")
    def validate_flight_type(cls, v):
        if v not in {"N", "I"}:
            raise ValueError('TIPOVUELO must be either "N" or "I"')
        return v


class FlightsRequest(BaseModel):
    flights: List[Flight]

    @validator("flights")
    def validate_flights_not_empty(cls, v):
        if not v:
            raise ValueError("flights list cannot be empty")
        return v


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(data: FlightsRequest) -> dict:
    try:
        # Entrenamiento bajo demanda por si el startup no se ejecutó en tests
        _ensure_trained()

        # Construir DataFrame con los campos que tu preprocess requiere
        flights_data = []
        for f in data.flights:
            flights_data.append(
                {
                    "OPERA": f.OPERA,
                    "TIPOVUELO": f.TIPOVUELO,
                    "MES": f.MES,
                    # Valores dummy requeridos por preprocess (fechas y varios campos)
                    "Fecha-I": "2022-01-01 10:00:00",
                    "Fecha-O": "2022-01-01 10:00:00",
                    "DIA": 1,
                    "AÑO": 2022,
                    "DIANOM": "Lunes",
                    "Vlo-I": "DUMMY",
                    "Ori-I": "DUMMY",
                    "Des-I": "DUMMY",
                    "Emp-I": "DUMMY",
                    "Vlo-O": "DUMMY",
                    "Ori-O": "DUMMY",
                    "Des-O": "DUMMY",
                    "Emp-O": "DUMMY",
                    "SIGLAORI": "DUMMY",
                    "SIGLADES": "DUMMY",
                }
            )

        df = pd.DataFrame(flights_data)
        X = model.preprocess(df)
        preds = model.predict(X)
        return {"predict": preds}

    except ValueError as e:
        # Errores de negocio -> 400 (lo que esperan tus tests)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Cualquier otro error inesperado
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
