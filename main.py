import numpy as np
import requests
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import tensorflow as tf
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load models and scalers
MODEL_VERSION = "v1.0"
LOAD_TIME = datetime.utcnow().isoformat()

try:
    occurrence_model = tf.keras.models.load_model("wildfire_occurrence_model.keras")
    severity_model = tf.keras.models.load_model("wildfire_severity_model.keras")
    occurrence_scaler = joblib.load("occurrence_scaler.joblib")
    severity_scaler = joblib.load("severity_scaler.joblib")
    logger.info("✅ Models and scalers loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load models or scalers: {e}")
    occurrence_model, severity_model, occurrence_scaler, severity_scaler = None, None, None, None

# FastAPI app
app = FastAPI(title="Wildfire Prediction API", version=MODEL_VERSION)

class PredictionRequest(BaseModel):
    lat: float
    lon: float
    date: str  # format: YYYY-MM-DD


# ============================================================
# SAFE HELPERS
# ============================================================

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


# ============================================================
# FWI (same as training)
# ============================================================

def calc_fwi_simple(temp_mean, wind_speed_mean, humidity_min):
    try:
        if humidity_min is None or humidity_min <= 0:
            return 0.0
        if temp_mean is None:
            temp_mean = 0.0
        if wind_speed_mean is None:
            wind_speed_mean = 0.0
        return (temp_mean * wind_speed_mean) / (humidity_min + 1)
    except Exception as e:
        logger.error(f"FWI calculation error: {e}")
        return 0.0


# ============================================================
# WEATHER FEATURE EXTRACTION
# ============================================================

def fetch_weather_features(lat, lon, date):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date,
            "end_date": date,
            "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,"
                      "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
                      "surface_pressure,precipitation,shortwave_radiation,"
                      "cloud_cover,et0_fao_evapotranspiration",
            "timezone": "UTC"
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Weather API error {response.status_code}")

        data = response.json().get("hourly", {})
        if not data:
            raise Exception("No hourly data returned from API")

        # Build features
        temp_vals = data.get("temperature_2m", [])
        humidity_vals = data.get("relative_humidity_2m", [])
        wind_vals = data.get("wind_speed_10m", [])
        wind_dir_vals = data.get("wind_direction_10m", [])

        features = {
            "temp_mean": safe_mean(temp_vals, "temperature"),
            "temp_max": safe_max(temp_vals, "temperature"),
            "temp_min": safe_min(temp_vals, "temperature"),
            "temp_range": safe_max(temp_vals, "temperature") - safe_min(temp_vals, "temperature"),
            "humidity_mean": safe_mean(humidity_vals, "humidity"),
            "humidity_min": safe_min(humidity_vals, "humidity"),
            "humidity_max": safe_max(humidity_vals, "humidity"),
            "dewpoint_mean": safe_mean(data.get("dewpoint_2m", []), "dewpoint"),
            "wind_speed_mean": safe_mean(wind_vals, "wind speed"),
            "wind_speed_max": safe_max(wind_vals, "wind speed"),
            "wind_gust_max": safe_max(data.get("wind_gusts_10m", []), "wind gusts"),
            "wind_direction_mean": safe_mean(wind_dir_vals, "wind direction"),
            "wind_direction_std": float(np.nanstd(wind_dir_vals)) if wind_dir_vals else 0.0,
            "pressure_mean": safe_mean(data.get("surface_pressure", []), "pressure"),
            "solar_radiation_mean": safe_mean(data.get("shortwave_radiation", []), "solar radiation"),
            "solar_radiation_max": safe_max(data.get("shortwave_radiation", []), "solar radiation"),
            "cloud_cover_mean": safe_mean(data.get("cloud_cover", []), "cloud cover"),
            "evapotranspiration_total": safe_sum(data.get("et0_fao_evapotranspiration", []), "evapotranspiration"),
        }

        # Derived FWI
        features["fire_weather_index"] = calc_fwi_simple(
            features["temp_mean"], features["wind_speed_mean"], features["humidity_min"]
        )

        return features

    except Exception as e:
        logger.error(f"❌ Failed to build features: {e}")
        raise


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

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
    try:
        if not all([occurrence_model, severity_model, occurrence_scaler, severity_scaler]):
            return {"detail": "Models not loaded"}

        # Weather features
        features = fetch_weather_features(request.lat, request.lon, request.date)

        # Cyclical encodings for month
        date_obj = datetime.strptime(request.date, "%Y-%m-%d")
        month = date_obj.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        # Lat/Lon encodings
        lat_rad = np.radians(request.lat)
        lon_rad = np.radians(request.lon)
        lat_sin, lat_cos = np.sin(lat_rad), np.cos(lat_rad)
        lon_sin, lon_cos = np.sin(lon_rad), np.cos(lon_rad)

        # Final feature vector (same order as training)
        feature_vector = np.array([[
            0,  # daynight_N (removed, always 0)
            features["fire_weather_index"],
            features["pressure_mean"],
            features["wind_direction_mean"],
            features["wind_direction_std"],
            features["solar_radiation_mean"],
            features["dewpoint_mean"],
            features["cloud_cover_mean"],
            features["evapotranspiration_total"],
            features["humidity_min"],
            features["temp_mean"],
            features["temp_range"],
            features["wind_speed_max"],
            lat_sin, lat_cos, lon_sin, lon_cos,
            month_sin, month_cos
        ]], dtype=float)

        # Occurrence prediction
        occ_features = occurrence_scaler.transform(feature_vector)
        occ_prob = float(occurrence_model.predict(occ_features)[0][0])
        occ_class = 1 if occ_prob >= 0.36 else 0

        # Severity prediction
        sev_features = severity_scaler.transform(feature_vector)
        sev_probs = severity_model.predict(sev_features)[0]
        sev_class = int(np.argmax(sev_probs))

        return {
            "occurrence_probability": occ_prob,
            "occurrence_class": occ_class,
            "severity_probabilities": sev_probs.tolist(),
            "severity_class": sev_class
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"detail": f"Prediction failed: {e}"}
