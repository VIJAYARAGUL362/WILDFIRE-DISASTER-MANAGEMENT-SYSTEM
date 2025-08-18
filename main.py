# app/main.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import numpy as np
import requests
from typing import Optional

# -------------------------
# Load models and scalers once at startup
# -------------------------
OCCURRENCE_MODEL_PATH = "wildfire_occurrence_model.keras"
OCCURRENCE_SCALER_PATH = "occurence_scaler.joblib"
SEVERITY_MODEL_PATH = "wildfire_severity_model.keras"
SEVERITY_SCALER_PATH = "severity_scaler.joblib"

occurrence_model = load_model(OCCURRENCE_MODEL_PATH)
occurrence_scaler = joblib.load(OCCURRENCE_SCALER_PATH)

severity_model = load_model(SEVERITY_MODEL_PATH)
severity_scaler = joblib.load(SEVERITY_SCALER_PATH)
SEVERITY_THRESHOLDS = [0.6, 0.6, 0.8]  # per-class thresholds

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Wildfire Prediction API", version="1.0")

# -------------------------
# Request model
# -------------------------
class PredictRequest(BaseModel):
    lat: float
    lon: float
    date: str  # "YYYY-MM-DD"

# -------------------------
# Utility functions (reuse from your scripts)
# -------------------------
def get_daynight_flag(hour=None):
    from datetime import datetime
    if hour is None:
        hour = datetime.now().hour
    return 0.0 if 6 <= hour < 18 else 1.0

def calc_fwi_quick(temp_c, rh, wind_kmh, rain_mm):
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

def fetch_weather(lat, lon, date):
    """Fetch weather data from Open-Meteo (hourly + daily)"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": ["temperature_2m","dewpoint_2m","relative_humidity_2m","pressure_msl",
                   "wind_speed_10m","wind_direction_10m","shortwave_radiation","cloudcover"],
        "daily": ["precipitation_sum","et0_fao_evapotranspiration"],
        "timezone": "auto"
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")

    hourly = data.get("hourly", {})
    daily = data.get("daily", {})

    # Convert to arrays
    def safe_mean(arr): return np.mean(arr) if len(arr)>0 else 0.0
    def safe_std(arr): return np.std(arr) if len(arr)>0 else 0.0
    def safe_max(arr): return np.max(arr) if len(arr)>0 else 0.0
    def safe_min(arr): return np.min(arr) if len(arr)>0 else 0.0
    def safe_sum(arr): return np.sum(arr) if len(arr)>0 else 0.0

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
        "evapotranspiration_total": daily.get("et0_fao_evapotranspiration",[0])[0],
        "humidity_min": safe_min(hourly.get("relative_humidity_2m", [])),
        "temp_mean": safe_mean(hourly.get("temperature_2m", [])),
        "temp_range": safe_max(hourly.get("temperature_2m", [])) - safe_min(hourly.get("temperature_2m", [])),
        "wind_speed_max": safe_max(hourly.get("wind_speed_10m", []))
    }
    return features

def build_features(lat, lon, date):
    weather = fetch_weather(lat, lon, date)
    dt = datetime.strptime(date, "%Y-%m-%d")
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    month_angle = 2*np.pi*(dt.month/12)

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
    return np.array(features).reshape(1,-1)

# -------------------------
# API Endpoint
# -------------------------
@app.post("/predict")
def predict(data: PredictRequest):
    lat, lon, date = data.lat, data.lon, data.date

    # --- Occurrence prediction ---
    X_occ = build_features(lat, lon, date)
    X_occ_scaled = occurrence_scaler.transform(X_occ)
    occ_probs = occurrence_model.predict(X_occ_scaled)[0]
    occ_class = int((occ_probs > 0.36).astype(int)[0])

    # --- Severity prediction ---
    X_sev = build_features(lat, lon, date)
    X_sev_scaled = severity_scaler.transform(X_sev)
    sev_probs = severity_model.predict(X_sev_scaled)[0]
    # Per-class threshold logic
    candidates = [i for i,p in enumerate(sev_probs) if p >= SEVERITY_THRESHOLDS[i]]
    if len(candidates)==0:
        sev_class = int(np.argmax(sev_probs))
    else:
        sev_class = max(candidates, key=lambda i: sev_probs[i])

    return {
        "occurrence": {
            "probs": float(occ_probs[0]),
            "predicted_class": occ_class
        },
        "severity": {
            "probs": sev_probs.tolist(),
            "predicted_class": sev_class,
            "thresholds": SEVERITY_THRESHOLDS
        }
    }
