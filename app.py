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
from datetime import datetime, timedelta
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
            # Check if date is reasonable - allow historical dates from 1940
            today = datetime.utcnow().date()
            days_diff = (dt.date() - today).days

            # Allow dates from 1940 onwards (ERA5 data starts from 1940)
            if dt.year < 1940:
                raise ValueError("Date cannot be earlier than 1940 (ERA5 data limitation)")
            if days_diff > 14:
                raise ValueError("Date cannot be more than 14 days in the future")
            return v
        except ValueError as e:
            if "Date cannot be" in str(e) or "ERA5" in str(e):
                raise e
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
def get_api_date_range():
    """Get the current valid date range for the weather API"""
    try:
        # Try a test request to get the current date range
        url = "https://api.open-meteo.com/v1/forecast?latitude=0&longitude=0&daily=temperature_2m_mean"
        r = requests.get(url, timeout=5)
        if r.status_code == 400 and "out of allowed range" in r.text:
            # Extract date range from error message if available
            # This is a fallback - in practice you might want to handle this differently
            pass

        # Default safe range - you should update this based on API documentation
        today = datetime.utcnow().date()
        start_date = today - timedelta(days=7)  # 7 days ago
        end_date = today + timedelta(days=7)  # 7 days ahead

        return start_date, end_date
    except:
        # Fallback range
        today = datetime.utcnow().date()
        return today - timedelta(days=7), today + timedelta(days=7)


def fetch_weather(lat, lon, date_str):
    """Fetch weather data from Open-Meteo API with better error handling"""
    # Determine if we need historical or forecast API
    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    today = datetime.utcnow().date()
    days_ago = (today - date_obj).days

    if days_ago > 5:  # Use historical API for dates more than 5 days ago
        api_endpoint = "https://archive-api.open-meteo.com/v1/era5"
    else:  # Use forecast API for recent/future dates
        api_endpoint = "https://api.open-meteo.com/v1/forecast"

    url = (
        f"{api_endpoint}"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"dewpoint_2m_mean,relative_humidity_2m_min,precipitation_sum,"
        f"windspeed_10m_max,winddirection_10m_dominant,shortwave_radiation_sum,"
        f"surface_pressure_mean,cloudcover_mean,et0_fao_evapotranspiration_sum"
        f"&timezone=UTC&start_date={date_str}&end_date={date_str}"
    )

    try:
        r = requests.get(url, timeout=10)

        # Handle specific API errors
        if r.status_code == 400:
            error_msg = r.text
            if "out of allowed range" in error_msg:
                # Try to get current date and suggest alternative
                today = datetime.utcnow().date()
                raise HTTPException(
                    status_code=400,
                    detail=f"Date {date_str} is outside the weather API's allowed range. "
                           f"Try using today's date: {today.strftime('%Y-%m-%d')} or check API documentation for valid date ranges."
                )
            else:
                raise HTTPException(status_code=400, detail=f"Weather API error: {error_msg}")

        r.raise_for_status()
        data = r.json()["daily"]

        # Check if we got data
        if not data or not data.get("temperature_2m_mean"):
            raise HTTPException(status_code=404, detail="No weather data available for the specified date and location")

        return {
            "temp_mean": data["temperature_2m_mean"][0] if data["temperature_2m_mean"][0] is not None else 20.0,
            "temp_range": (data["temperature_2m_max"][0] - data["temperature_2m_min"][0]) if all(
                x is not None for x in [data["temperature_2m_max"][0], data["temperature_2m_min"][0]]) else 10.0,
            "dewpoint_mean": data["dewpoint_2m_mean"][0] if data["dewpoint_2m_mean"][0] is not None else 15.0,
            "humidity_min": data["relative_humidity_2m_min"][0] if data["relative_humidity_2m_min"][
                                                                       0] is not None else 50.0,
            "wind_speed_max": data["windspeed_10m_max"][0] if data["windspeed_10m_max"][0] is not None else 5.0,
            "wind_direction_mean": data["winddirection_10m_dominant"][0] if data["winddirection_10m_dominant"][
                                                                                0] is not None else 180.0,
            "wind_direction_std": np.random.uniform(0, 20),  # placeholder
            "solar_radiation_mean": data["shortwave_radiation_sum"][0] if data["shortwave_radiation_sum"][
                                                                              0] is not None else 150.0,
            "pressure_mean": data["surface_pressure_mean"][0] if data["surface_pressure_mean"][
                                                                     0] is not None else 1013.0,
            "cloud_cover_mean": data["cloudcover_mean"][0] if data["cloudcover_mean"][0] is not None else 50.0,
            "evapotranspiration_total": data["et0_fao_evapotranspiration_sum"][0] if
            data["et0_fao_evapotranspiration_sum"][0] is not None else 3.0,
            "fire_weather_index": (
                    (data["temperature_2m_mean"][0] * (100 - data["relative_humidity_2m_min"][0]))
                    / (data["precipitation_sum"][0] + 1)
            ) if all(x is not None for x in [data["temperature_2m_mean"][0], data["relative_humidity_2m_min"][0],
                                             data["precipitation_sum"][0]]) else 5.0
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Weather data fetch failed: {str(e)}. Please check your coordinates and date."
        )


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

    X = df[features]
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


@app.api_route("/health", methods=["GET", "HEAD"])
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


@app.get("/api-info")
def api_info():
    """Get current API date range information"""
    try:
        today = datetime.utcnow().date()
        return {
            "current_date": today.strftime('%Y-%m-%d'),
            "suggested_date": today.strftime('%Y-%m-%d'),
            "note": "Use today's date or check weather API documentation for valid date ranges"
        }
    except Exception as e:
        return {"error": str(e)}


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

    except HTTPException:
        # Re-raise HTTP exceptions (like weather API errors)
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


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
            "/model/info": "Model information",
            "/api-info": "Get current valid date information"
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