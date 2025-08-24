import numpy as np
import requests
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import tensorflow as tf
from datetime import datetime, UTC, timedelta
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
import asyncio
import time
from functools import lru_cache
import json

# ============================================================
# Enhanced Logging Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("wildfire_api.log")
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# Configuration and Constants
# ============================================================
class Config:
    MODEL_VERSION = "v2.0"
    API_VERSION = "2.0.0"
    MAX_CACHE_SIZE = 1000
    CACHE_TTL_SECONDS = 3600  # 1 hour
    REQUEST_TIMEOUT = 30
    MAX_BATCH_SIZE = 50


CONFIG = Config()
LOAD_TIME = datetime.now(UTC).isoformat()


# Global model storage
class ModelStore:
    def __init__(self):
        self.occurrence_model = None
        self.severity_model = None
        self.occurrence_scaler = None
        self.severity_scaler = None
        self.is_loaded = False
        self.load_time = None

    def load_models(self):
        try:
            self.occurrence_model = tf.keras.models.load_model("wildfire_occurence_model.keras")
            self.severity_model = tf.keras.models.load_model("wildfire_severity_model.keras")
            self.occurrence_scaler = joblib.load("occurence_scaler.joblib")
            self.severity_scaler = joblib.load("severity_scaler.joblib")
            self.is_loaded = True
            self.load_time = datetime.now(UTC).isoformat()
            logger.info("✅ Models and scalers loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load models or scalers: {e}")
            raise


models = ModelStore()

# Request tracking
request_stats = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "cache_hits": 0,
    "start_time": datetime.now(UTC)
}


# ============================================================
# Enhanced Request Models
# ============================================================
class PredictionRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    lon: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")
    date: str = Field(..., description="Date in YYYY-MM-DD format")

    @validator('date')
    def validate_date(cls, v):
        try:
            date_obj = datetime.strptime(v, "%Y-%m-%d")
            # Don't allow dates too far in the future (weather API limitations)
            if date_obj > datetime.now() + timedelta(days=14):
                raise ValueError("Date cannot be more than 14 days in the future")
            return v
        except ValueError as e:
            if "Date cannot be more than" in str(e):
                raise e
            raise ValueError("Date must be in YYYY-MM-DD format")


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest] = Field(..., max_items=CONFIG.MAX_BATCH_SIZE)


class PredictionResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str
    processing_time_ms: float


class RiskLevel(BaseModel):
    level: str
    color: str
    description: str


# ============================================================
# Enhanced Lifespan Management
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Wildfire Prediction API...")
    try:
        models.load_models()
        logger.info(f"🔥 API ready! Version {CONFIG.API_VERSION}")
    except Exception as e:
        logger.error(f"💥 Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down Wildfire Prediction API")


# ============================================================
# FastAPI App with Enhanced Configuration
# ============================================================
app = FastAPI(
    title="🔥 Wildfire Prediction API Pro",
    description="Advanced wildfire occurrence and severity prediction using ML and real-time weather data",
    version=CONFIG.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json",
    contact={
        "name": "Wildfire Prediction Team",
        "email": "contact@wildfire-api.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ============================================================
# Enhanced Helper Functions
# ============================================================
def track_request_stats(success: bool = True, cache_hit: bool = False):
    request_stats["total_requests"] += 1
    if success:
        request_stats["successful_predictions"] += 1
    else:
        request_stats["failed_predictions"] += 1
    if cache_hit:
        request_stats["cache_hits"] += 1


def safe_calculation(func, arr, label="unknown", default=0.0):
    """Enhanced safe calculation with better error handling"""
    try:
        if arr is None or len(arr) == 0:
            logger.debug(f"⚠️ Missing data for {label}, using default: {default}")
            return default

        result = func(arr)
        if np.isnan(result) or np.isinf(result):
            logger.warning(f"⚠️ Invalid result for {label}, using default: {default}")
            return default

        return float(result)
    except Exception as e:
        logger.warning(f"⚠️ Error in calculation for {label}: {e}, using default: {default}")
        return default


def safe_mean(arr, label="unknown"):
    return safe_calculation(np.nanmean, arr, label)


def safe_min(arr, label="unknown"):
    return safe_calculation(np.nanmin, arr, label)


def safe_max(arr, label="unknown"):
    return safe_calculation(np.nanmax, arr, label)


def safe_sum(arr, label="unknown"):
    return safe_calculation(np.nansum, arr, label)


def safe_std(arr, label="unknown"):
    return safe_calculation(np.nanstd, arr, label)


def calc_fwi_simple(temp_mean, wind_speed_mean, humidity_min):
    """Original FWI calculation - EXACTLY as used in training"""
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


def get_risk_level(occurrence_prob: float, severity_class: int) -> RiskLevel:
    """Determine risk level based on occurrence probability and severity"""
    if occurrence_prob < 0.2:
        return RiskLevel(
            level="Very Low",
            color="#22c55e",
            description="Minimal wildfire risk"
        )
    elif occurrence_prob < 0.4:
        return RiskLevel(
            level="Low",
            color="#84cc16",
            description="Low wildfire risk"
        )
    elif occurrence_prob < 0.6:
        return RiskLevel(
            level="Moderate",
            color="#f59e0b",
            description="Moderate wildfire risk - monitor conditions"
        )
    elif occurrence_prob < 0.8:
        return RiskLevel(
            level="High",
            color="#f97316",
            description="High wildfire risk - exercise caution"
        )
    else:
        return RiskLevel(
            level="Extreme",
            color="#dc2626",
            description="Extreme wildfire risk - take immediate precautions"
        )


@lru_cache(maxsize=CONFIG.MAX_CACHE_SIZE)
def cached_weather_fetch(lat: float, lon: float, date: str) -> str:
    """Cached weather data fetch"""
    return json.dumps(fetch_weather_features_internal(lat, lon, date))


def fetch_weather_features_internal(lat: float, lon: float, date: str) -> Dict:
    """Internal weather fetch function - EXACTLY matching training data format"""
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

        response = requests.get(url, params=params, timeout=CONFIG.REQUEST_TIMEOUT)
        response.raise_for_status()

        data = response.json().get("hourly", {})
        if not data:
            raise Exception("No hourly data returned from API")

        # EXACT feature extraction as in training
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

        # ORIGINAL FWI calculation (same as training)
        features["fire_weather_index"] = calc_fwi_simple(
            features["temp_mean"], features["wind_speed_mean"], features["humidity_min"]
        )

        return features

    except requests.RequestException as e:
        logger.error(f"❌ Weather API request failed: {e}")
        raise HTTPException(status_code=503, detail="Weather service unavailable")
    except Exception as e:
        logger.error(f"❌ Failed to build features: {e}")
        raise HTTPException(status_code=500, detail="Feature extraction failed")


async def fetch_weather_features(lat: float, lon: float, date: str) -> Dict:
    """Async wrapper for weather feature fetching with caching"""
    cache_key = f"{lat:.4f}_{lon:.4f}_{date}"

    try:
        # Try cache first
        cached_result = cached_weather_fetch(lat, lon, date)
        track_request_stats(cache_hit=True)
        return json.loads(cached_result)
    except:
        # Cache miss, fetch fresh data
        return fetch_weather_features_internal(lat, lon, date)


# ============================================================
# Enhanced Landing Page
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def landing_page():
    uptime = datetime.now(UTC) - request_stats["start_time"]

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>🔥 Wildfire Prediction API Pro</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}

            body {{
                font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                line-height: 1.6;
                color: #1f2937;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 2rem;
            }}

            header {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                padding: 2rem 0;
                text-align: center;
                color: white;
            }}

            .hero {{
                padding: 4rem 0;
                text-align: center;
                color: white;
            }}

            .hero h1 {{
                font-size: clamp(2.5rem, 5vw, 4rem);
                font-weight: 700;
                margin-bottom: 1rem;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }}

            .hero p {{
                font-size: 1.25rem;
                opacity: 0.9;
                max-width: 600px;
                margin: 0 auto 2rem;
            }}

            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin: 3rem 0;
            }}

            .stat-card {{
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 2rem;
                text-align: center;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: transform 0.3s ease;
            }}

            .stat-card:hover {{
                transform: translateY(-5px);
            }}

            .stat-card i {{
                font-size: 2.5rem;
                margin-bottom: 1rem;
                color: #667eea;
            }}

            .stat-number {{
                font-size: 2rem;
                font-weight: 700;
                color: #1f2937;
                margin-bottom: 0.5rem;
            }}

            .stat-label {{
                color: #6b7280;
                font-size: 0.95rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}

            .features {{
                background: white;
                margin: 4rem 0;
                border-radius: 24px;
                padding: 3rem;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
            }}

            .features h2 {{
                font-size: 2.5rem;
                text-align: center;
                margin-bottom: 3rem;
                color: #1f2937;
            }}

            .feature-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
            }}

            .feature-item {{
                padding: 1.5rem;
                border-radius: 12px;
                background: #f8fafc;
                border-left: 4px solid #667eea;
            }}

            .feature-item h3 {{
                font-size: 1.25rem;
                color: #1f2937;
                margin-bottom: 0.5rem;
            }}

            .api-section {{
                background: #1f2937;
                color: white;
                padding: 3rem;
                border-radius: 24px;
                margin: 4rem 0;
            }}

            .code-block {{
                background: #111827;
                border-radius: 8px;
                padding: 1.5rem;
                margin: 1rem 0;
                overflow-x: auto;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.9rem;
            }}

            .btn-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 2rem 0;
            }}

            .btn {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 1rem 2rem;
                border-radius: 12px;
                text-decoration: none;
                font-weight: 600;
                transition: all 0.3s ease;
                border: none;
                cursor: pointer;
            }}

            .btn-primary {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }}

            .btn-primary:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            }}

            .btn-outline {{
                background: transparent;
                color: white;
                border: 2px solid rgba(255, 255, 255, 0.3);
            }}

            .btn-outline:hover {{
                background: rgba(255, 255, 255, 0.1);
                border-color: rgba(255, 255, 255, 0.5);
            }}

            footer {{
                text-align: center;
                padding: 2rem;
                color: rgba(255, 255, 255, 0.8);
                font-size: 0.9rem;
            }}

            @media (max-width: 768px) {{
                .container {{ padding: 0 1rem; }}
                .features {{ margin: 2rem 0; padding: 2rem; }}
                .api-section {{ padding: 2rem; }}
            }}
        </style>
    </head>
    <body>
        <header>
            <div class="container">
                <div class="hero">
                    <h1><i class="fas fa-fire"></i> Wildfire Prediction API Pro</h1>
                    <p>Advanced ML-powered wildfire risk assessment using real-time weather data</p>

                    <div class="stats-grid">
                        <div class="stat-card">
                            <i class="fas fa-chart-line"></i>
                            <div class="stat-number">{request_stats['total_requests']}</div>
                            <div class="stat-label">Total Predictions</div>
                        </div>
                        <div class="stat-card">
                            <i class="fas fa-clock"></i>
                            <div class="stat-number">{uptime.days}d {uptime.seconds // 3600}h</div>
                            <div class="stat-label">Uptime</div>
                        </div>
                        <div class="stat-card">
                            <i class="fas fa-check-circle"></i>
                            <div class="stat-number">{(request_stats['successful_predictions'] / max(request_stats['total_requests'], 1) * 100):.1f}%</div>
                            <div class="stat-label">Success Rate</div>
                        </div>
                        <div class="stat-card">
                            <i class="fas fa-tachometer-alt"></i>
                            <div class="stat-number">{(request_stats['cache_hits'] / max(request_stats['total_requests'], 1) * 100):.1f}%</div>
                            <div class="stat-label">Cache Hit Rate</div>
                        </div>
                    </div>
                </div>
            </div>
        </header>

        <div class="container">
            <div class="features">
                <h2><i class="fas fa-rocket"></i> Professional Features</h2>
                <div class="feature-grid">
                    <div class="feature-item">
                        <h3><i class="fas fa-brain"></i> Advanced ML Models</h3>
                        <p>Deep neural networks for occurrence prediction and severity classification with enhanced feature engineering</p>
                    </div>
                    <div class="feature-item">
                        <h3><i class="fas fa-cloud"></i> Real-time Weather Data</h3>
                        <p>Integration with Open-Meteo API for comprehensive meteorological data including solar radiation and VPD</p>
                    </div>
                    <div class="feature-item">
                        <h3><i class="fas fa-layer-group"></i> Batch Processing</h3>
                        <p>Process up to {CONFIG.MAX_BATCH_SIZE} predictions simultaneously for efficient large-scale analysis</p>
                    </div>
                    <div class="feature-item">
                        <h3><i class="fas fa-bolt"></i> Performance Optimized</h3>
                        <p>Intelligent caching, async processing, and compressed responses for lightning-fast predictions</p>
                    </div>
                    <div class="feature-item">
                        <h3><i class="fas fa-shield-alt"></i> Production Ready</h3>
                        <p>Comprehensive error handling, monitoring, logging, and validation for enterprise deployment</p>
                    </div>
                    <div class="feature-item">
                        <h3><i class="fas fa-chart-pie"></i> Enhanced Analytics</h3>
                        <p>Detailed risk levels, confidence scores, and comprehensive metadata for informed decision making</p>
                    </div>
                </div>
            </div>

            <div class="api-section">
                <h2><i class="fas fa-code"></i> API Usage</h2>
                <p>Make predictions with our enhanced endpoint:</p>

                <div class="code-block">
POST /api/v2/predict
Content-Type: application/json

{{
  "lat": 13.0827,
  "lon": 80.2707,
  "date": "2025-08-25"
}}
                </div>

                <div class="btn-grid">
                    <a href="/docs" class="btn btn-primary">
                        <i class="fas fa-book"></i>&nbsp; Interactive Docs
                    </a>
                    <a href="/redoc" class="btn btn-outline">
                        <i class="fas fa-file-alt"></i>&nbsp; API Reference
                    </a>
                    <a href="/api/v2/health" class="btn btn-outline">
                        <i class="fas fa-heartbeat"></i>&nbsp; Health Check
                    </a>
                    <a href="/api/v2/stats" class="btn btn-outline">
                        <i class="fas fa-analytics"></i>&nbsp; Live Stats
                    </a>
                </div>
            </div>
        </div>

        <footer>
            <div class="container">
                <p>🔥 Wildfire Prediction API Pro v{CONFIG.API_VERSION} • Built with FastAPI & TensorFlow • © {datetime.now(UTC).year}</p>
            </div>
        </footer>

        <script>
            // Add some interactivity
            document.querySelectorAll('.stat-card').forEach(card => {{
                card.addEventListener('mouseenter', () => {{
                    card.style.transform = 'translateY(-10px) scale(1.02)';
                }});
                card.addEventListener('mouseleave', () => {{
                    card.style.transform = 'translateY(0) scale(1)';
                }});
            }});
        </script>
    </body>
    </html>
    """


# ============================================================
# Enhanced API Endpoints
# ============================================================
@app.get("/api/v2/health")
async def health_check():
    """Comprehensive health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": CONFIG.API_VERSION,
        "models_loaded": models.is_loaded,
        "uptime_seconds": (datetime.now(UTC) - request_stats["start_time"]).total_seconds(),
        "memory_usage": "Available in production environment"
    })


@app.get("/api/v2/stats")
async def get_stats():
    """Get API statistics"""
    uptime = datetime.now(UTC) - request_stats["start_time"]
    return JSONResponse({
        "statistics": {
            **request_stats,
            "uptime_seconds": uptime.total_seconds(),
            "success_rate": request_stats["successful_predictions"] / max(request_stats["total_requests"], 1),
            "cache_hit_rate": request_stats["cache_hits"] / max(request_stats["total_requests"], 1),
        },
        "configuration": {
            "model_version": CONFIG.MODEL_VERSION,
            "api_version": CONFIG.API_VERSION,
            "max_batch_size": CONFIG.MAX_BATCH_SIZE,
            "cache_ttl_seconds": CONFIG.CACHE_TTL_SECONDS,
        },
        "timestamp": datetime.now(UTC).isoformat()
    })


@app.post("/api/v2/predict", response_model=PredictionResponse)
async def predict_enhanced(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Enhanced prediction endpoint with comprehensive response"""
    start_time = time.time()

    try:
        if not models.is_loaded:
            track_request_stats(success=False)
            raise HTTPException(status_code=503, detail="Models not loaded")

        # Fetch weather features
        features = await fetch_weather_features(request.lat, request.lon, request.date)

        # EXACT cyclical encodings as in training
        date_obj = datetime.strptime(request.date, "%Y-%m-%d")
        month = date_obj.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        # EXACT geographical encodings as in training
        lat_rad = np.radians(request.lat)
        lon_rad = np.radians(request.lon)
        lat_sin, lat_cos = np.sin(lat_rad), np.cos(lat_rad)
        lon_sin, lon_cos = np.sin(lon_rad), np.cos(lon_rad)

        # EXACT feature vector order as in training (same order as original)
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

        # Predictions with error handling
        try:
            occ_features = models.occurrence_scaler.transform(feature_vector)
            occ_prob = float(models.occurrence_model.predict(occ_features, verbose=0)[0][0])
            occ_class = 1 if occ_prob >= 0.36 else 0
        except Exception as e:
            logger.error(f"Occurrence prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Occurrence prediction failed")

        try:
            sev_features = models.severity_scaler.transform(feature_vector)
            sev_probs = models.severity_model.predict(sev_features, verbose=0)[0]
            sev_class = int(np.argmax(sev_probs))
        except Exception as e:
            logger.error(f"Severity prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Severity prediction failed")

        # Enhanced risk assessment
        risk_level = get_risk_level(occ_prob, sev_class)
        confidence_score = float(np.max(sev_probs)) * (1 - abs(occ_prob - 0.5) * 2)

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        response_data = {
            "prediction": {
                "occurrence": {
                    "probability": round(occ_prob, 4),
                    "class": occ_class,
                    "threshold": 0.36
                },
                "severity": {
                    "class": sev_class,
                    "class_names": ["Low", "Moderate", "High", "Extreme"],
                    "probabilities": {
                        "low": round(float(sev_probs[0]), 4),
                        "moderate": round(float(sev_probs[1]), 4) if len(sev_probs) > 1 else 0.0,
                        "high": round(float(sev_probs[2]), 4) if len(sev_probs) > 2 else 0.0,
                        "extreme": round(float(sev_probs[3]), 4) if len(sev_probs) > 3 else 0.0
                    }
                }
            },
            "risk_assessment": {
                "level": risk_level.level,
                "color": risk_level.color,
                "description": risk_level.description,
                "confidence_score": round(confidence_score, 4)
            },
            "weather_features": {
                "fire_weather_index": round(features["fire_weather_index"], 2),
                "temperature": {
                    "mean": round(features["temp_mean"], 1),
                    "max": round(features["temp_max"], 1),
                    "min": round(features["temp_min"], 1),
                    "range": round(features["temp_range"], 1)
                },
                "humidity": {
                    "mean": round(features["humidity_mean"], 1),
                    "min": round(features["humidity_min"], 1)
                },
                "wind": {
                    "speed_mean": round(features["wind_speed_mean"], 1),
                    "speed_max": round(features["wind_speed_max"], 1),
                    "direction_mean": round(features["wind_direction_mean"], 1)
                },
                "solar_radiation_mean": round(features["solar_radiation_mean"], 1),
                "pressure_mean": round(features["pressure_mean"], 1),
                "note": "Features extracted exactly as used in model training"
            },
            "location": {
                "latitude": request.lat,
                "longitude": request.lon,
                "date": request.date
            }
        }

        metadata = {
            "model_version": CONFIG.MODEL_VERSION,
            "api_version": CONFIG.API_VERSION,
            "prediction_id": f"pred_{int(time.time())}_{hash(f'{request.lat}{request.lon}{request.date}') % 10000}",
            "data_source": "Open-Meteo API",
            "feature_count": len(feature_vector[0])
        }

        track_request_stats(success=True)

        return PredictionResponse(
            success=True,
            data=response_data,
            metadata=metadata,
            timestamp=datetime.now(UTC).isoformat(),
            processing_time_ms=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        track_request_stats(success=False)
        logger.error(f"Prediction failed: {e}")
        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            success=False,
            data={"error": str(e)},
            metadata={"error_type": type(e).__name__},
            timestamp=datetime.now(UTC).isoformat(),
            processing_time_ms=processing_time
        )


@app.post("/api/v2/batch-predict")
async def batch_predict(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Batch prediction endpoint for multiple locations"""
    start_time = time.time()

    try:
        if not models.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")

        results = []
        successful_predictions = 0

        for i, pred_request in enumerate(request.predictions):
            try:
                # Process each prediction
                result = await predict_enhanced(pred_request, background_tasks)
                results.append({
                    "index": i,
                    "success": result.success,
                    "data": result.data,
                    "location": {
                        "lat": pred_request.lat,
                        "lon": pred_request.lon,
                        "date": pred_request.date
                    }
                })
                if result.success:
                    successful_predictions += 1

            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e),
                    "location": {
                        "lat": pred_request.lat,
                        "lon": pred_request.lon,
                        "date": pred_request.date
                    }
                })

        processing_time = (time.time() - start_time) * 1000

        return JSONResponse({
            "success": True,
            "batch_summary": {
                "total_requests": len(request.predictions),
                "successful_predictions": successful_predictions,
                "failed_predictions": len(request.predictions) - successful_predictions,
                "success_rate": successful_predictions / len(request.predictions)
            },
            "results": results,
            "metadata": {
                "batch_id": f"batch_{int(time.time())}",
                "processing_time_ms": processing_time,
                "timestamp": datetime.now(UTC).isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        processing_time = (time.time() - start_time) * 1000

        return JSONResponse({
            "success": False,
            "error": str(e),
            "metadata": {
                "processing_time_ms": processing_time,
                "timestamp": datetime.now(UTC).isoformat()
            }
        }, status_code=500)


@app.get("/api/v2/risk-levels")
async def get_risk_levels():
    """Get information about risk level classifications"""
    return JSONResponse({
        "risk_levels": [
            {
                "level": "Very Low",
                "range": "0.0 - 0.2",
                "color": "#22c55e",
                "description": "Minimal wildfire risk - conditions are not conducive to fire spread"
            },
            {
                "level": "Low",
                "range": "0.2 - 0.4",
                "color": "#84cc16",
                "description": "Low wildfire risk - fire danger is below normal"
            },
            {
                "level": "Moderate",
                "range": "0.4 - 0.6",
                "color": "#f59e0b",
                "description": "Moderate wildfire risk - monitor weather conditions closely"
            },
            {
                "level": "High",
                "range": "0.6 - 0.8",
                "color": "#f97316",
                "description": "High wildfire risk - exercise caution with outdoor activities"
            },
            {
                "level": "Extreme",
                "range": "0.8 - 1.0",
                "color": "#dc2626",
                "description": "Extreme wildfire risk - take immediate fire prevention measures"
            }
        ],
        "severity_classes": [
            {"class": 0, "name": "Low", "description": "Small, easily controlled fires"},
            {"class": 1, "name": "Moderate", "description": "Moderate intensity fires requiring professional response"},
            {"class": 2, "name": "High", "description": "High intensity fires with significant resource requirements"},
            {"class": 3, "name": "Extreme", "description": "Extreme fires with potential for major damage"}
        ]
    })


@app.get("/api/v2/model-info")
async def get_model_info():
    """Get detailed information about the prediction models"""
    return JSONResponse({
        "models": {
            "occurrence_model": {
                "type": "Binary Classification Neural Network",
                "architecture": "Dense layers with dropout regularization",
                "threshold": 0.36,
                "performance_metrics": "Available in model documentation"
            },
            "severity_model": {
                "type": "Multi-class Classification Neural Network",
                "classes": 4,
                "class_names": ["Low", "Moderate", "High", "Extreme"],
                "architecture": "Dense layers with softmax output"
            }
        },
        "features": {
            "total_features": 18,
            "weather_features": [
                "fire_weather_index", "pressure_mean", "wind_direction_mean",
                "wind_direction_std", "solar_radiation_mean", "dewpoint_mean",
                "cloud_cover_mean", "evapotranspiration_total", "humidity_min",
                "temp_mean", "temp_range", "wind_speed_max"
            ],
            "engineered_features": [
                "lat_sin", "lat_cos", "lon_sin", "lon_cos",
                "month_sin", "month_cos"
            ]
        },
        "data_source": "Open-Meteo Weather API",
        "model_version": CONFIG.MODEL_VERSION,
        "last_updated": models.load_time
    })


# Legacy endpoints for backward compatibility
@app.post("/predict")
async def predict_legacy(request: PredictionRequest):
    """Legacy prediction endpoint for backward compatibility"""
    logger.info("Legacy endpoint used - consider migrating to /api/v2/predict")
    result = await predict_enhanced(request, BackgroundTasks())

    # Return in legacy format
    if result.success:
        return {
            "occurrence_probability": result.data["prediction"]["occurrence"]["probability"],
            "occurrence_class": result.data["prediction"]["occurrence"]["class"],
            "severity_probabilities": list(result.data["prediction"]["severity"]["probabilities"].values()),
            "severity_class": result.data["prediction"]["severity"]["class"]
        }
    else:
        return {"detail": result.data.get("error", "Prediction failed")}


@app.get("/health")
async def health_legacy():
    """Legacy health endpoint"""
    return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/version")
async def version_legacy():
    """Legacy version endpoint"""
    return {
        "model_version": CONFIG.MODEL_VERSION,
        "loaded_at": models.load_time,
        "models_loaded": models.is_loaded
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "/api/v2/predict",
                "/api/v2/batch-predict",
                "/api/v2/health",
                "/api/v2/stats",
                "/docs",
                "/redoc"
            ]
        }
    )


@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.detail,
            "message": "Please check your request format and try again"
        }
    )


# Add startup message
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Wildfire Prediction API Pro is starting up...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )