# wildfire_api.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import logging
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import requests
from datetime import datetime
from typing import List, Dict, Optional
import os

# ------------------------
# Logging configuration
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------
# FastAPI initialization
# ------------------------
app = FastAPI(
    title="Wildfire Prediction API",
    description="Predict wildfire occurrence and severity with enhanced endpoints",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Load models and scalers
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models
occurrence_model = load_model(os.path.join(BASE_DIR, "wildfire_occurrence_model.h5"))
severity_model = load_model(os.path.join(BASE_DIR, "wildfire_severity_model.h5"))

# Scalers
occurrence_scaler = joblib.load(os.path.join(BASE_DIR, "scaler_occured.joblib"))
severity_scaler = joblib.load(os.path.join(BASE_DIR, "scaler_severity.joblib"))

# ------------------------
# Request / Response Models
# ------------------------
class PredictionRequest(BaseModel):
    lat: float = Field(..., description="Latitude in decimal degrees")
    lon: float = Field(..., description="Longitude in decimal degrees")
    date: str = Field(..., description="Date in YYYY-MM-DD format")

    @validator("date")
    def validate_date(cls, v):
        try:
            dt = datetime.strptime(v, "%Y-%m-%d")
            today = datetime.utcnow().date()
            if (dt.date() - today).days > 14:
                raise ValueError("Date cannot be more than 14 days in the future")
            return v
        except Exception:
            raise ValueError("Invalid date format, expected YYYY-MM-DD")

class DirectPredictionRequest(BaseModel):
    cloud_cover_mean: float
    dewpoint_mean: float
    evapotranspiration_total: float
    fire_weather_index: float
    humidity_min: float
    pressure_mean: float
    solar_radiation_mean: float
    temp_mean: float
    temp_range: float
    wind_direction_mean: float
    wind_direction_std: float
    wind_speed_max: float
    daynight_N: int
    lat: float
    lon: float
    month: int

class PredictionResponse(BaseModel):
    occurrence_probability: float
    occurrence_class: int
    severity_probabilities: List[float]
    severity_class: int


# ------------------------
# Weather API
# ------------------------
def fetch_weather(lat, lon, date_str):
    """Fetch weather data from Open-Meteo API"""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"dewpoint_2m_mean,relative_humidity_2m_min,precipitation_sum,"
        f"windspeed_10m_max,winddirection_10m_dominant,shortwave_radiation_sum,"
        f"surface_pressure_mean,cloudcover_mean,evapotranspiration_sum"
        f"&timezone=UTC&start_date={date_str}&end_date={date_str}"
    )

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()["daily"]

        return {
            "temp_mean": data["temperature_2m_mean"][0],
            "temp_range": data["temperature_2m_max"][0] - data["temperature_2m_min"][0],
            "dewpoint_mean": data["dewpoint_2m_mean"][0],
            "humidity_min": data["relative_humidity_2m_min"][0],
            "wind_speed_max": data["windspeed_10m_max"][0],
            "wind_direction_mean": data["winddirection_10m_dominant"][0],
            "wind_direction_std": np.random.uniform(0, 20),  # placeholder
            "solar_radiation_mean": data["shortwave_radiation_sum"][0],
            "pressure_mean": data["surface_pressure_mean"][0],
            "cloud_cover_mean": data["cloudcover_mean"][0],
            "evapotranspiration_total": data["evapotranspiration_sum"][0],
            "fire_weather_index": (
                (data["temperature_2m_mean"][0] * (100 - data["relative_humidity_2m_min"][0]))
                / (data["precipitation_sum"][0] + 1)
            )
        }
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        raise HTTPException(status_code=500, detail="Weather data fetch failed")

# ------------------------
# Core Prediction Functions
# ------------------------
def predict_fire_risk(input_data):
    """Core fire risk prediction function"""
    df = pd.DataFrame([input_data])

    # Feature engineering
    df["temp_humidity_index"] = df["temp_mean"] * (100 - df["humidity_min"])
    df["temp_dewpoint_diff"] = df["temp_mean"] - df["dewpoint_mean"]
    df["fwi_wind_ratio"] = df["fire_weather_index"] / (df["wind_speed_max"] + 0.1)
    df["radiation_evapo_product"] = df["solar_radiation_mean"] * df["evapotranspiration_total"]

    # Encode cyclical features
    df["lat_cos"] = np.cos(np.radians(df["lat"]))
    df["lat_sin"] = np.sin(np.radians(df["lat"]))
    df["lon_cos"] = np.cos(np.radians(df["lon"]))
    df["lon_sin"] = np.sin(np.radians(df["lon"]))
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)

    # Feature order
    features = [
        "cloud_cover_mean", "dewpoint_mean", "evapotranspiration_total",
        "fire_weather_index", "humidity_min", "pressure_mean",
        "solar_radiation_mean", "temp_mean", "temp_range",
        "wind_direction_mean", "wind_direction_std", "wind_speed_max",
        "temp_humidity_index", "temp_dewpoint_diff", "fwi_wind_ratio",
        "radiation_evapo_product", "daynight_N", "lat_cos", "lat_sin",
        "lon_cos", "lon_sin", "month_cos", "month_sin",
    ]

    X = df[features].values

    # Occurrence prediction
    X_occ_scaled = occurrence_scaler.transform(X)
    occ_prob = float(occurrence_model.predict(X_occ_scaled)[0][0])
    occ_class = int(occ_prob > 0.5)

    # Severity prediction
    X_sev_scaled = severity_scaler.transform(X)
    sev_probs = severity_model.predict(X_sev_scaled)[0]
    sev_probs_list = [float(p) for p in sev_probs]
    sev_class = int(np.argmax(sev_probs))

    return {
        "occurrence_probability": occ_prob,
        "occurrence_class": occ_class,
        "severity_probabilities": sev_probs_list,
        "severity_class": sev_class,
    }

def get_risk_interpretation(occurrence_prob, severity_class):
    """Interpret risk levels for better understanding"""
    if occurrence_prob < 0.3:
        risk_level = "LOW"
    elif occurrence_prob < 0.6:
        risk_level = "MODERATE" if severity_class < 2 else "HIGH"
    else:
        risk_level = "HIGH" if severity_class < 2 else "CRITICAL"
    
    severity_labels = ["Low", "Moderate", "High"]
    severity_label = severity_labels[severity_class]
    
    recommendations = {
        "LOW": "Normal fire safety precautions recommended.",
        "MODERATE": "Increased vigilance advised. Monitor conditions.",
        "HIGH": "High fire danger. Avoid outdoor burning. Be prepared.",
        "CRITICAL": "Extreme fire danger. Emergency preparedness required."
    }
    
    return {
        "risk_level": risk_level,
        "severity_label": severity_label,
        "recommendation": recommendations[risk_level],
        "occurrence_percentage": f"{occurrence_prob * 100:.1f}%"
    }

def get_daynight_flag():
    hour = datetime.utcnow().hour
    return 0 if 6 <= hour < 18 else 1

# ------------------------
# API Endpoints
# ------------------------
@app.get("/")
def root():
    return {"message": "Wildfire Prediction API", "version": "2.2.0", "status": "running"}

@app.get("/health")
def health_check():
    """API health check endpoint"""
    try:
        test_data = {
            "cloud_cover_mean": 50.0, "dewpoint_mean": 15.0,
            "evapotranspiration_total": 2.0, "fire_weather_index": 3.0,
            "humidity_min": 30.0, "pressure_mean": 1013.0,
            "solar_radiation_mean": 150.0, "temp_mean": 25.0,
            "temp_range": 8.0, "wind_direction_mean": 90.0,
            "wind_direction_std": 10.0, "wind_speed_max": 8.0,
            "daynight_N": 1, "lat": 20.0, "lon": 80.0, "month": 6
        }
        predict_fire_risk(test_data)
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
def predict_with_coords(req: PredictionRequest):
    """Predict using coordinates and date (fetches weather data)"""
    try:
        weather = fetch_weather(req.lat, req.lon, req.date)
        dt = datetime.strptime(req.date, "%Y-%m-%d")
        
        input_data = {
            **weather,
            "daynight_N": get_daynight_flag(),
            "lat": req.lat,
            "lon": req.lon,
            "month": dt.month
        }
        
        result = predict_fire_risk(input_data)
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/direct", response_model=PredictionResponse)
def predict_direct(req: DirectPredictionRequest):
    """Direct prediction with provided weather data"""
    try:
        input_data = req.dict()
        result = predict_fire_risk(input_data)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Direct prediction error: {e}")
        raise HTTPException(status_code=500, detail="Direct prediction failed")



@app.get("/model/info")
def get_model_info():
    """Return model information and required features"""
    return {
        "model_version": "1.0",
        "occurrence_model": "Binary classification (fire/no fire)",
        "severity_model": "3-class classification (Low/Moderate/High)",
        "required_features": [
            "cloud_cover_mean", "dewpoint_mean", "evapotranspiration_total",
            "fire_weather_index", "humidity_min", "pressure_mean",
            "solar_radiation_mean", "temp_mean", "temp_range",
            "wind_direction_mean", "wind_direction_std", "wind_speed_max",
            "daynight_N", "lat", "lon", "month"
        ],
        "endpoints": {
            "/predict": "Prediction using coordinates and date",
            "/predict/direct": "Direct prediction with provided data",
            "/predict/enhanced": "Enhanced prediction with interpretation",
            "/predict/batch": "Batch prediction for multiple locations",
            "/health": "Health check",
            "/model/info": "Model information"
        }
    }

@app.get("/date-info")
def date_info(date: str):
    """Get date information"""
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        return {
            "day": dt.day, 
            "month": dt.month, 
            "year": dt.year, 
            "weekday": dt.strftime("%A")
        }
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)