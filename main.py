from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import requests
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# FastAPI app with CORS
# -------------------------
app = FastAPI(
    title="Wildfire Prediction API",
    description="AI-powered wildfire occurrence and severity prediction system",
    version="1.0"
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
# Global variables for models
# -------------------------
occurrence_model = None
occurrence_scaler = None
severity_model = None
severity_scaler = None
SEVERITY_THRESHOLDS = [0.6, 0.6, 0.8]


# -------------------------
# Load models function
# -------------------------
def load_models():
    global occurrence_model, occurrence_scaler, severity_model, severity_scaler
    try:
        # Try to import tensorflow
        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            logger.error("TensorFlow not installed")
            return False

        occurrence_model = load_model("wildfire_occurrence_model.keras")
        occurrence_scaler = joblib.load("occurence_scaler.joblib")
        severity_model = load_model("wildfire_severity_model.keras")
        severity_scaler = joblib.load("severity_scaler.joblib")
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False


# -------------------------
# Request/Response models
# -------------------------
class PredictRequest(BaseModel):
    lat: float
    lon: float
    date: str


class PredictionResponse(BaseModel):
    occurrence_probability: float
    occurrence_class: int
    severity_probabilities: list
    severity_class: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# -------------------------
# Utility functions
# -------------------------
def get_daynight_flag(hour=None):
    if hour is None:
        hour = datetime.now().hour
    return 0.0 if 6 <= hour < 18 else 1.0


def calc_fwi_quick(temp_c, rh, wind_kmh, rain_mm):
    try:
        rh = float(np.clip(rh, 1e-3, 100))
        wind_kmh = max(0.0, float(wind_kmh))
        rain_mm = max(0.0, float(rain_mm or 0.0))

        mo = 147.2 * (101.0 - rh) / (59.5 + rh)
        ffmc = np.clip(101.0 - mo + 0.6 * (temp_c - 20.0), 0.0, 101.0)
        wind_factor = np.exp(0.05039 * wind_kmh)
        isi = 0.208 * wind_factor * max(0.0, ffmc - 80.0)
        bui = max(0.0, temp_c * (1 - rh / 100.0) - 0.3 * rain_mm)
        fwi = isi * np.exp(0.023 * bui)
        return float(max(0.0, fwi))
    except Exception as e:
        logger.error(f"FWI calculation error: {e}")
        return 0.0


def fetch_weather(lat, lon, date_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ["temperature_2m", "dewpoint_2m", "relative_humidity_2m", "pressure_msl",
                   "wind_speed_10m", "wind_direction_10m", "shortwave_radiation", "cloudcover"],
        "daily": ["precipitation_sum", "et0_fao_evapotranspiration"],
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")

    hourly = data.get("hourly", {})
    daily = data.get("daily", {})

    def safe_mean(arr):
        return np.mean(arr) if arr and len(arr) > 0 else 0.0

    def safe_std(arr):
        return np.std(arr) if arr and len(arr) > 0 else 0.0

    def safe_max(arr):
        return np.max(arr) if arr and len(arr) > 0 else 0.0

    def safe_min(arr):
        return np.min(arr) if arr and len(arr) > 0 else 0.0

    def safe_sum(arr):
        return np.sum(arr) if arr and len(arr) > 0 else 0.0

    fwi = calc_fwi_quick(
        temp_c=safe_mean(hourly.get("temperature_2m", [])),
        rh=safe_min(hourly.get("relative_humidity_2m", [])),
        wind_kmh=safe_max(hourly.get("wind_speed_10m", [])),
        rain_mm=safe_sum(daily.get("precipitation_sum", [0.0]))
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
        raise HTTPException(status_code=500, detail="Failed to build features")


# -------------------------
# API Endpoints
# -------------------------
@app.get("/")
async def root():
    return {
        "message": "Wildfire Prediction API",
        "version": "1.0",
        "endpoints": ["/predict", "/health", "/docs"]
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    models_loaded = all([
        occurrence_model is not None,
        occurrence_scaler is not None,
        severity_model is not None,
        severity_scaler is not None
    ])

    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictRequest):
    # Check if models are loaded
    if not all([occurrence_model, occurrence_scaler, severity_model, severity_scaler]):
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Validate date format
    try:
        datetime.strptime(data.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    try:
        # Build features
        X = build_features(data.lat, data.lon, data.date)

        # Occurrence prediction
        X_occ_scaled = occurrence_scaler.transform(X)
        occ_probs = occurrence_model.predict(X_occ_scaled)[0]
        occ_prob = float(occ_probs[0])
        occ_class = int(occ_prob > 0.36)

        # Severity prediction
        X_sev_scaled = severity_scaler.transform(X)
        sev_probs = severity_model.predict(X_sev_scaled)[0]
        sev_probs_list = [float(p) for p in sev_probs]

        # Per-class threshold logic
        candidates = [i for i, p in enumerate(sev_probs) if p >= SEVERITY_THRESHOLDS[i]]
        if len(candidates) == 0:
            sev_class = int(np.argmax(sev_probs))
        else:
            sev_class = max(candidates, key=lambda i: sev_probs[i])

        return PredictionResponse(
            occurrence_probability=occ_prob,
            occurrence_class=occ_class,
            severity_probabilities=sev_probs_list,
            severity_class=sev_class
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


# -------------------------
# Startup event
# -------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Wildfire Prediction API...")
    success = load_models()
    if success:
        logger.info("API ready!")
    else:
        logger.warning("API started but models failed to load")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )