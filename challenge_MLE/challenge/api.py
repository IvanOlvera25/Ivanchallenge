import fastapi
import pandas as pd
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST
from pydantic import BaseModel, validator
from typing import List
from pathlib import Path
from challenge.model import DelayModel

app = fastapi.FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"detail": exc.errors()})

model = DelayModel()

@app.on_event("startup")
async def startup_event():
    try:
        data_file = Path(__file__).resolve().parents[1] / "data" / "data.csv"
        data = pd.read_csv(data_file)
        features, target = model.preprocess(data, target_column='delay')
        model.fit(features, target)
        print("Model trained successfully!")
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise e

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator('MES')
    def validate_month(cls, v):
        if not 1 <= v <= 12:
            raise ValueError('MES must be between 1 and 12')
        return v

    @validator('TIPOVUELO')
    def validate_flight_type(cls, v):
        if v not in ['N', 'I']:
            raise ValueError('TIPOVUELO must be either "N" or "I"')
        return v

class FlightPredictionInput(BaseModel):
    flights: List[Flight]

    @validator('flights')
    def validate_flights_not_empty(cls, v):
        if not v:
            raise ValueError('flights list cannot be empty')
        return v

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(data: FlightPredictionInput) -> dict:
    try:
        flights_data = []
        for flight in data.flights:
            flights_data.append({
                'OPERA': flight.OPERA,
                'TIPOVUELO': flight.TIPOVUELO,
                'MES': flight.MES,
                'Fecha-I': '2022-01-01 10:00:00',
                'Fecha-O': '2022-01-01 10:00:00',
                'DIA': 1, 'AÑO': 2022, 'DIANOM': 'Lunes',
                'Vlo-I': 'DUMMY', 'Ori-I': 'DUMMY', 'Des-I': 'DUMMY', 'Emp-I': 'DUMMY',
                'Vlo-O': 'DUMMY', 'Ori-O': 'DUMMY', 'Des-O': 'DUMMY', 'Emp-O': 'DUMMY',
                'SIGLAORI': 'DUMMY', 'SIGLADES': 'DUMMY'
            })
        df = pd.DataFrame(flights_data)
        features = model.preprocess(df)
        predictions = model.predict(features)
        return {"predict": predictions}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
