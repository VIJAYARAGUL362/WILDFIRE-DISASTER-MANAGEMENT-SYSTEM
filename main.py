import os
import numpy as np
import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import tensorflow as tf
from datetime import datetime

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wildfire_api")

# ---------------- App ----------------
MODEL_VERSION = "v1.0"
LOAD_TIME = datetime.utcnow().isoformat()

app = FastAPI(
    title="Wildfire Prediction API",
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    root_path=os.getenv("ROOT_PATH", "")  # helps if Railway proxies under a subpath
)

# ---------------- Globals (set on startup) ----------------
occurrence_model = None
severity_model = None
occurrence_scaler = None
severity_scaler = None

class PredictionRequest(BaseModel):
    lat: float
    lon: float
    date: str  # YYYY-MM-DD

# ---------------- Safe helpers ----------------
def safe_mean(arr, label="unknown"):
    try:
        if arr is None or len(arr) == 0:
            logger.warning(f"⚠️ Missing data for {label}, defaulting to 0.0")
            return 0.0
        return float(np.nanmean(arr))
    except Exception:
        logger.warning(f"⚠️ Error in mean calc for {label}, defaulting to 0.0")
        return 0.0

def safe_min(arr, label="unknown"):
    try:
        if arr is None or len(arr) == 0:
            logger.warning(f"⚠️ Missing data for {label}, defaulting to 0.0")
            return 0.0
        return float(np.nanmin(arr))
    except Exception:
        logger.warning(f"⚠️ Error in min calc for {label}, defaulting to 0.0")
        return 0.0

def safe_max(arr, label="unknown"):
    try:
        if arr is None or len(arr) == 0:
            logger.warning(f"⚠️ Missing data for {label}, defaulting to 0.0")
            return 0.0
        return float(np.nanmax(arr))
    except Exception:
        logger.warning(f"⚠️ Error in max calc for {label}, defaulting to 0.0")
        return 0.0

def safe_sum(arr, label="unknown"):
    try:
        if arr is None or len(arr) == 0:
            logger.warning(f"⚠️ Missing data for {label}, defaulting to 0.0")
            return 0.0
        return float(np.nansum(arr))
    except Exception:
        logger.warning(f"⚠️ Error in sum calc for {label}, defaulting to 0.0")
        return 0.0

def calc_fwi_simple(temp_mean, wind_speed_mean, humidity_min):
    try:
        if humidity_min is None or humidity_min <= 0:
            return 0.0
        if temp_mean is None:
            temp_mean = 0.0
        if wind_speed_mean is None:
            wind_speed_mean = 0.0
        return float((temp_mean * wind_speed_mean) / (humidity_min + 1))
    except Exception as e:
        logger.error(f"FWI calculation error: {e}")
        return 0.0

# ---------------- Weather features ----------------
def fetch_weather_features(lat, lon, date):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date,
            "end_date": date,
            "hourly": (
                "temperature_2m,relative_humidity_2m,dewpoint_2m,"
                "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
                "surface_pressure,precipitation,shortwave_radiation,"
                "cloud_cover,et0_fao_evapotranspiration"
            ),
            "timezone": "UTC"
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Weather API error {r.status_code}")

        hourly = r.json().get("hourly", {})
        if not hourly:
            raise RuntimeError("No hourly data returned from API")

        temp_vals = hourly.get("temperature_2m", [])
        humidity_vals = hourly.get("relative_humidity_2m", [])
        wind_vals = hourly.get("wind_speed_10m", [])
        wind_dir_vals = hourly.get("wind_direction_10m", [])

        feats = {
            "temp_mean": safe_mean(temp_vals, "temperature"),
            "temp_max": safe_max(temp_vals, "temperature"),
            "temp_min": safe_min(temp_vals, "temperature"),
            "temp_range": safe_max(temp_vals, "temperature") - safe_min(temp_vals, "temperature"),
            "humidity_mean": safe_mean(humidity_vals, "humidity"),
            "humidity_min": safe_min(humidity_vals, "humidity"),
            "humidity_max": safe_max(humidity_vals, "humidity"),
            "dewpoint_mean": safe_mean(hourly.get("dewpoint_2m", []), "dewpoint"),
            "wind_speed_mean": safe_mean(wind_vals, "wind speed"),
            "wind_speed_max": safe_max(wind_vals, "wind speed"),
            "wind_gust_max": safe_max(hourly.get("wind_gusts_10m", []), "wind gusts"),
            "wind_direction_mean": safe_mean(wind_dir_vals, "wind direction"),
            "wind_direction_std": float(np.nanstd(wind_dir_vals)) if wind_dir_vals else 0.0,
            "pressure_mean": safe_mean(hourly.get("surface_pressure", []), "pressure"),
            "solar_radiation_mean": safe_mean(hourly.get("shortwave_radiation", []), "solar radiation"),
            "solar_radiation_max": safe_max(hourly.get("shortwave_radiation", []), "solar radiation"),
            "cloud_cover_mean": safe_mean(hourly.get("cloud_cover", []), "cloud cover"),
            "evapotranspiration_total": safe_sum(hourly.get("et0_fao_evapotranspiration", []), "evapotranspiration"),
        }

        feats["fire_weather_index"] = calc_fwi_simple(
            feats["temp_mean"], feats["wind_speed_mean"], feats["humidity_min"]
        )
        return feats
    except Exception as e:
        logger.error(f"❌ Failed to build features: {e}")
        raise

# ---------------- Startup: load models/scalers ----------------
@app.on_event("startup")
def load_artifacts():
    global occurrence_model, severity_model, occurrence_scaler, severity_scaler
    try:
        occurrence_model = tf.keras.models.load_model("wildfire_occurrence_model.keras")
        severity_model = tf.keras.models.load_model("wildfire_severity_model.keras")
        occurrence_scaler = joblib.load("occurrence_scaler.joblib")
        severity_scaler = joblib.load("severity_scaler.joblib")
        logger.info("✅ Models and scalers loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load models/scalers at startup: {e}")

# ---------------- Endpoints ----------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": all([occurrence_model, severity_model, occurrence_scaler, severity_scaler]),
        "version": MODEL_VERSION
    }

@app.get("/ping")
def ping():
    return {"ping": "pong"}

@app.get("/version")
def version():
    return {
        "model_version": MODEL_VERSION,
        "loaded_at": LOAD_TIME,
        "models_loaded": all([occurrence_model, severity_model, occurrence_scaler, severity_scaler])
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    if not all([occurrence_model, severity_model, occurrence_scaler, severity_scaler]):
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        feats = fetch_weather_features(request.lat, request.lon, request.date)

        # Cyclical encodings
        date_obj = datetime.strptime(request.date, "%Y-%m-%d")
        month = date_obj.month
        month_sin = float(np.sin(2 * np.pi * month / 12))
        month_cos = float(np.cos(2 * np.pi * month / 12))

        # Lat/Lon encodings
        lat_rad = np.radians(request.lat)
        lon_rad = np.radians(request.lon)
        lat_sin, lat_cos = float(np.sin(lat_rad)), float(np.cos(lat_rad))
        lon_sin, lon_cos = float(np.sin(lon_rad)), float(np.cos(lon_rad))

        # Feature vector (same order you trained)
        X = np.array([[
            0,  # daynight_N
            feats["fire_weather_index"],
            feats["pressure_mean"],
            feats["wind_direction_mean"],
            feats["wind_direction_std"],
            feats["solar_radiation_mean"],
            feats["dewpoint_mean"],
            feats["cloud_cover_mean"],
            feats["evapotranspiration_total"],
            feats["humidity_min"],
            feats["temp_mean"],
            feats["temp_range"],
            feats["wind_speed_max"],
            lat_sin, lat_cos, lon_sin, lon_cos,
            month_sin, month_cos
        ]], dtype=float)

        X_occ = occurrence_scaler.transform(X)
        occ_prob = float(occurrence_model.predict(X_occ)[0][0])
        occ_class = int(occ_prob >= 0.36)

        X_sev = severity_scaler.transform(X)
        sev_probs = severity_model.predict(X_sev)[0]
        sev_probs = [float(p) for p in sev_probs]
        sev_class = int(np.argmax(sev_probs))

        return {
            "occurrence_probability": occ_prob,
            "occurrence_class": occ_class,
            "severity_probabilities": sev_probs,
            "severity_class": sev_class
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
