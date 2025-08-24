import fastapi
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, validator
from typing import List
from challenge.model import DelayModel

app = fastapi.FastAPI()

# Initialize model
model = DelayModel()

# Load and train the model at startup
@app.on_event("startup")
async def startup_event():
    """Load and train the model when the API starts."""
    try:
        # Load training data
        data = pd.read_csv('./data/data.csv')
        
        # Preprocess and train
        features, target = model.preprocess(data, target_column='delay')
        model.fit(features, target)
        
        print("Model trained successfully!")
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise e


class Flight(BaseModel):
    """Input model for flight data."""
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
    """Input model for prediction request."""
    flights: List[Flight]
    
    @validator('flights')
    def validate_flights_not_empty(cls, v):
        if not v:
            raise ValueError('flights list cannot be empty')
        return v


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(data: FlightPredictionInput) -> dict:
    try:
        # Convert input data to DataFrame
        flights_data = []
        for flight in data.flights:
            # Create a dummy row with required columns for preprocessing
            flight_dict = {
                'OPERA': flight.OPERA,
                'TIPOVUELO': flight.TIPOVUELO,
                'MES': flight.MES,
                # Add dummy values for required preprocessing columns
                'Fecha-I': '2022-01-01 10:00:00',  # Dummy scheduled time
                'Fecha-O': '2022-01-01 10:00:00',  # Dummy operation time
                'DIA': 1,
                'AÑO': 2022,
                'DIANOM': 'Lunes',
                'Vlo-I': 'DUMMY',
                'Ori-I': 'DUMMY',
                'Des-I': 'DUMMY',
                'Emp-I': 'DUMMY',
                'Vlo-O': 'DUMMY',
                'Ori-O': 'DUMMY',
                'Des-O': 'DUMMY',
                'Emp-O': 'DUMMY',
                'SIGLAORI': 'DUMMY',
                'SIGLADES': 'DUMMY'
            }
            flights_data.append(flight_dict)
        
        df = pd.DataFrame(flights_data)
        
        # Preprocess features
        features = model.preprocess(df)
        
        # Make predictions
        predictions = model.predict(features)
        
        return {"predict": predictions}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")