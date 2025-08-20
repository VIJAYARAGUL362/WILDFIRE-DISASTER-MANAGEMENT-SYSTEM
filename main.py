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


def calc_fwi_quick(temp_c, rh, wind_kmh, rain_mm):
    try:
        logger.info(f"Calculating FWI with: temp={temp_c}, rh={rh}, wind={wind_kmh}, rain={rain_mm}")

        rh = float(np.clip(rh, 1e-3, 100))
        wind_kmh = max(0.0, float(wind_kmh))
        rain_mm = max(0.0, float(rain_mm or 0.0))

        mo = 147.2 * (101.0 - rh) / (59.5 + rh)
        ffmc = np.clip(101.0 - mo + 0.6 * (temp_c - 20.0), 0.0, 101.0)
        wind_factor = np.exp(0.05039 * wind_kmh)
        isi = 0.208 * wind_factor * max(0.0, ffmc - 80.0)
        bui = max(0.0, temp_c * (1 - rh / 100.0) - 0.3 * rain_mm)
        fwi = isi * np.exp(0.023 * bui)

        result = float(max(0.0, fwi))
        logger.info(f"Calculated FWI: {result}")
        return result
    except Exception as e:
        logger.error(f"FWI calculation error: {e}")
        return 0.0


def fetch_weather(lat, lon, date_str):
    logger.info(f"Fetching weather for lat={lat}, lon={lon}, date={date_str}")

    # Determine if we need historical or forecast data
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    today = datetime.now().date()

    if target_date <= today:
        # Use historical weather API for past dates
        url = "https://archive-api.open-meteo.com/v1/archive"
        logger.info(f"Using historical weather API for date: {date_str}")
    else:
        # Use forecast API for future dates
        url = "https://api.open-meteo.com/v1/forecast"
        logger.info(f"Using forecast weather API for date: {date_str}")

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
        logger.info(f"Making request to: {url} with params: {params}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Weather API response received successfully")

        # Log the structure of the response for debugging
        logger.info(f"Response keys: {list(data.keys())}")
        if "hourly" in data:
            logger.info(f"Hourly keys: {list(data['hourly'].keys())}")
        if "daily" in data:
            logger.info(f"Daily keys: {list(data['daily'].keys())}")

    except requests.RequestException as e:
        logger.error(f"Weather API request error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")

            # Check if it's a date range error and provide helpful message
            if e.response.status_code == 400 and "out of allowed range" in e.response.text:
                raise HTTPException(
                    status_code=400,
                    detail=f"Date {date_str} is outside the available weather data range. Please use a date within the last few years for historical data or check the forecast range for future dates."
                )
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected weather API error: {e}")
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")

    hourly = data.get("hourly", {})
    daily = data.get("daily", {})

    def safe_mean(arr):
        if not arr or len(arr) == 0:
            logger.warning(f"Empty array for mean calculation")
            return 0.0
        try:
            result = np.mean(arr)
            return float(result) if not np.isnan(result) else 0.0
        except Exception as e:
            logger.error(f"Error calculating mean: {e}")
            return 0.0

    def safe_std(arr):
        if not arr or len(arr) == 0:
            logger.warning(f"Empty array for std calculation")
            return 0.0
        try:
            result = np.std(arr)
            return float(result) if not np.isnan(result) else 0.0
        except Exception as e:
            logger.error(f"Error calculating std: {e}")
            return 0.0

    def safe_max(arr):
        if not arr or len(arr) == 0:
            logger.warning(f"Empty array for max calculation")
            return 0.0
        try:
            result = np.max(arr)
            return float(result) if not np.isnan(result) else 0.0
        except Exception as e:
            logger.error(f"Error calculating max: {e}")
            return 0.0

    def safe_min(arr):
        if not arr or len(arr) == 0:
            logger.warning(f"Empty array for min calculation")
            return 0.0
        try:
            result = np.min(arr)
            return float(result) if not np.isnan(result) else 0.0
        except Exception as e:
            logger.error(f"Error calculating min: {e}")
            return 0.0

    def safe_sum(arr):
        if not arr or len(arr) == 0:
            logger.warning(f"Empty array for sum calculation")
            return 0.0
        try:
            result = np.sum(arr)
            return float(result) if not np.isnan(result) else 0.0
        except Exception as e:
            logger.error(f"Error calculating sum: {e}")
            return 0.0

    try:
        # Extract weather data with logging
        temp_data = hourly.get("temperature_2m", [])
        humidity_data = hourly.get("relative_humidity_2m", [])
        wind_data = hourly.get("wind_speed_10m", [])
        precip_data = daily.get("precipitation_sum", [0.0])

        logger.info(f"Temperature data length: {len(temp_data)}")
        logger.info(f"Humidity data length: {len(humidity_data)}")
        logger.info(f"Wind data length: {len(wind_data)}")
        logger.info(f"Precipitation data: {precip_data}")

        fwi = calc_fwi_quick(
            temp_c=safe_mean(temp_data),
            rh=safe_min(humidity_data),
            wind_kmh=safe_max(wind_data),
            rain_mm=safe_sum(precip_data)
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

        logger.info(f"Calculated features: {features}")
        return features

    except Exception as e:
        logger.error(f"Error processing weather data: {e}")
        raise


def build_features(lat, lon, date_str):
    try:
        logger.info(f"Building features for lat={lat}, lon={lon}, date={date_str}")

        # Validate inputs
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Invalid longitude: {lon}")

        # Parse date
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as e:
            logger.error(f"Date parsing error: {e}")
            raise ValueError(f"Invalid date format: {date_str}")

        # Fetch weather data
        weather = fetch_weather(lat, lon, date_str)

        # Calculate geographic features
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

        logger.info(f"Feature vector length: {len(features)}")
        logger.info(f"Feature values: {features}")

        # Check for NaN or inf values
        features_array = np.array(features)
        if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
            logger.error(f"Invalid values in features: {features}")
            raise ValueError("Features contain NaN or infinite values")

        return features_array.reshape(1, -1)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature building error: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to build features: {str(e)}")


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

    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded
    )


@app.get("/date-info", response_model=DateRangeResponse)
async def get_date_info():
    """Get information about available date ranges for weather data"""
    today = datetime.now().date()
    return DateRangeResponse(
        current_date=today.strftime("%Y-%m-%d"),
        historical_data_available="Available from 1940-01-01 to yesterday",
        forecast_range_info="Forecast range varies but typically 7-16 days from today. Check API response for exact range."
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


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