from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import joblib
import numpy as np
import requests
from datetime import datetime, timedelta
import logging
import os
from contextlib import asynccontextmanager
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Global variables for models
# -------------------------
occurrence_model = None
occurrence_scaler = None
severity_model = None
severity_scaler = None
SEVERITY_THRESHOLDS = [0.6, 0.6, 0.8]



FEATURE_NAMES = [
    "daynight_flag", "fire_weather_index", "pressure_mean",
    "wind_direction_mean", "wind_direction_std", "solar_radiation_mean",
    "dewpoint_mean", "cloud_cover_mean", "evapotranspiration_total",
    "humidity_min", "temp_mean", "temp_range", "wind_speed_max",
    "lat_sin", "lat_cos", "lon_sin", "lon_cos", "month_sin", "month_cos"
]


# -------------------------
# Load models function
# -------------------------
def load_models():
    global occurrence_model, occurrence_scaler, severity_model, severity_scaler
    try:
        from tensorflow.keras.models import load_model
        occurrence_model = load_model("wildfire_occurrence_model.keras")
        occurrence_scaler = joblib.load("occurence_scaler.joblib")
        severity_model = load_model("wildfire_severity_model.keras")
        severity_scaler = joblib.load("severity_scaler.joblib")
        logger.info("✅ Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return False

# -------------------------
# Request/Response models
# -------------------------
class PredictRequest(BaseModel):
    lat: float
    lon: float
    date: str

    @field_validator("date")
    def validate_date(cls, v):
        try:
            date_obj = datetime.strptime(v, "%Y-%m-%d").date()
            if date_obj > datetime.now().date() + timedelta(days=14):
                raise ValueError("Date cannot be more than 14 days in the future")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

class PredictionResponse(BaseModel):
    occurrence_probability: float
    occurrence_class: int
    severity_probabilities: list
    severity_class: int

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool

class DateRangeResponse(BaseModel):
    current_date: str
    historical_data_available: str
    forecast_range_info: str

# -------------------------
# Utility functions
# -------------------------
def get_daynight_flag(hour=None):
    if hour is None:
        hour = datetime.now().hour
    return 0.0 if 6 <= hour < 18 else 1.0

def calc_fwi_simple(temp_mean, wind_speed_mean, humidity_min):
    """Simple FWI formula used during dataset creation"""
    try:
        if humidity_min is None or humidity_min <= 0:
            return 0.0
        return (temp_mean * wind_speed_mean) / (humidity_min + 1)
    except Exception as e:
        logger.error(f"FWI calculation error: {e}")
        return 0.0

def fetch_weather(lat, lon, date_str):
    logger.info(f"Fetching weather for lat={lat}, lon={lon}, date={date_str}")

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    today = datetime.now().date()

    if target_date <= today:
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": [
            "temperature_2m", "dewpoint_2m", "relative_humidity_2m", "pressure_msl",
            "wind_speed_10m", "wind_direction_10m", "shortwave_radiation", "cloudcover"
        ],
        "daily": ["et0_fao_evapotranspiration"],
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")

    hourly = data.get("hourly", {})
    daily = data.get("daily", {})

    def safe_mean(arr): return float(np.mean(arr)) if arr else 0.0
    def safe_std(arr): return float(np.std(arr)) if arr else 0.0
    def safe_max(arr): return float(np.max(arr)) if arr else 0.0
    def safe_min(arr): return float(np.min(arr)) if arr else 0.0
    def safe_sum(arr): return float(np.sum(arr)) if arr else 0.0

    try:
        temp_data = hourly.get("temperature_2m", [])
        humidity_data = hourly.get("relative_humidity_2m", [])
        wind_data = hourly.get("wind_speed_10m", [])

        fwi = calc_fwi_simple(
            temp_mean=safe_mean(temp_data),
            wind_speed_mean=safe_mean(wind_data),
            humidity_min=safe_min(humidity_data)
        )

        features = {
            "fire_weather_index": fwi,
            "pressure_mean": safe_mean(hourly.get("pressure_msl", [])),
            "wind_direction_mean": safe_mean(hourly.get("wind_direction_10m", [])),
            "wind_direction_std": safe_std(hourly.get("wind_direction_10m", [])),
            "solar_radiation_mean": safe_mean(hourly.get("shortwave_radiation", [])),
            "dewpoint_mean": safe_mean(hourly.get("dewpoint_2m", [])),
            "cloud_cover_mean": safe_mean(hourly.get("cloudcover", [])),
            "evapotranspiration_total": safe_sum(daily.get("et0_fao_evapotranspiration", [0])),
            "humidity_min": safe_min(hourly.get("relative_humidity_2m", [])),
            "temp_mean": safe_mean(hourly.get("temperature_2m", [])),
            "temp_range": safe_max(hourly.get("temperature_2m", [])) - safe_min(hourly.get("temperature_2m", [])),
            "wind_speed_max": safe_max(hourly.get("wind_speed_10m", []))
        }
        return features
    except Exception as e:
        logger.error(f"Error processing weather data: {e}")
        raise

def build_features(lat, lon, date_str):
    try:
        weather = fetch_weather(lat, lon, date_str)
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        lat_rad, lon_rad = np.radians(lat), np.radians(lon)
        month_angle = 2 * np.pi * (dt.month / 12)

        features = [
            get_daynight_flag(),
            weather["fire_weather_index"],
            weather["pressure_mean"],
            weather["wind_direction_mean"],
            weather["wind_direction_std"],
            weather["solar_radiation_mean"],
            weather["dewpoint_mean"],
            weather["cloud_cover_mean"],
            weather["evapotranspiration_total"],
            weather["humidity_min"],
            weather["temp_mean"],
            weather["temp_range"],
            weather["wind_speed_max"],
            np.sin(lat_rad), np.cos(lat_rad),
            np.sin(lon_rad), np.cos(lon_rad),
            np.sin(month_angle), np.cos(month_angle)
        ]

        return np.array(features).reshape(1, -1)
    except Exception as e:
        logger.error(f"Feature building error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to build features: {str(e)}")

# -------------------------
# Lifespan management (replaces @on_event)
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Wildfire Prediction API...")
    load_models()
    yield
    logger.info("🛑 Shutting down Wildfire Prediction API")

# -------------------------
# FastAPI app with CORS
# -------------------------
app = FastAPI(
    title="Wildfire Prediction API",
    description="AI-powered wildfire occurrence and severity prediction system",
    version="1.0",
    lifespan=lifespan
)

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# API Endpoints
# -------------------------
@app.get("/")
async def root():
    return {
        "message": "Wildfire Prediction API",
        "version": "1.0",
        "endpoints": ["/predict", "/health", "/date-info", "/docs"]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    models_loaded = all([
        occurrence_model is not None,
        occurrence_scaler is not None,
        severity_model is not None,
        severity_scaler is not None
    ])
    return HealthResponse(status="healthy" if models_loaded else "unhealthy", models_loaded=models_loaded)

@app.get("/date-info", response_model=DateRangeResponse)
async def get_date_info():
    today = datetime.now().date()
    return DateRangeResponse(
        current_date=today.strftime("%Y-%m-%d"),
        historical_data_available="Available from 1940-01-01 to yesterday",
        forecast_range_info="Forecast range varies but typically 7-16 days from today. Check API response for exact range."
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictRequest):
    if not all([occurrence_model, occurrence_scaler, severity_model, severity_scaler]):
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        X = build_features(data.lat, data.lon, data.date)

        # Convert X into a DataFrame with feature names (fixes sklearn warning)
        X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

        # Occurrence prediction
        X_occ_scaled = occurrence_scaler.transform(X_df)
        occ_probs = occurrence_model.predict(X_occ_scaled)[0]
        occ_prob = float(occ_probs[0])
        occ_class = int(occ_prob > 0.50)

        # Severity prediction
        X_sev_scaled = severity_scaler.transform(X_df)
        sev_probs = severity_model.predict(X_sev_scaled)[0]
        sev_probs_list = [float(p) for p in sev_probs]

        candidates = [i for i, p in enumerate(sev_probs) if p >= SEVERITY_THRESHOLDS[i]]
        sev_class = int(np.argmax(sev_probs)) if len(candidates) == 0 else max(candidates, key=lambda i: sev_probs[i])

        return PredictionResponse(
            occurrence_probability=occ_prob,
            occurrence_class=occ_class,
            severity_probabilities=sev_probs_list,
            severity_class=sev_class
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
