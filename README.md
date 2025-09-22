# 🔥 Wildfire Prediction API

A sophisticated machine learning API built with FastAPI that predicts wildfire occurrence and severity using real-time weather data and advanced neural network models. This project demonstrates end-to-end ML engineering with production-ready deployment capabilities.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.2-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Dual Prediction Models**: Binary classification for fire occurrence + multi-class classification for severity
- **Real-time Weather Integration**: Automatic weather data fetching from Open-Meteo API
- **Advanced Feature Engineering**: Cyclical encoding, interaction features, and domain-specific indices
- **Multiple Prediction Modes**: Coordinate-based prediction and direct input prediction
- **Production Ready**: Dockerized deployment with comprehensive error handling
- **Interactive Documentation**: Auto-generated OpenAPI/Swagger documentation
- **Health Monitoring**: Built-in health checks and logging

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                     │
├─────────────────────────────────────────────────────────────┤
│  Endpoints:                                                 │
│  • /predict - Location + Date → Weather → Prediction       │
│  • /predict/direct - Direct weather data → Prediction      │
│  • /health - Health monitoring                             │
│  • /model/info - Model specifications                      │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                   Weather Data Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│  Open-Meteo API → Feature Engineering → Model Input        │
│  • Temperature, humidity, wind data                        │
│  • Fire Weather Index calculation                          │
│  • Cyclical encoding (lat/lon/month)                       │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML Models                               │
├─────────────────────────────────────────────────────────────┤
│  Occurrence Model (Binary):     Severity Model (3-class):  │
│  • TensorFlow/Keras Neural Net  • Low/Moderate/High        │
│  • StandardScaler preprocessing • Probability distribution │
│  • Binary classification        • Multi-class output       │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/wildfire-prediction-api.git
cd wildfire-prediction-api
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
# Development server
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. **Access the API**
- API Base URL: `http://localhost:8000`
- Interactive Documentation: `http://localhost:8000/docs`
- Alternative Documentation: `http://localhost:8000/redoc`

### Docker Deployment

```bash
# Build the Docker image
docker build -t wildfire-api .

# Run the container
docker run -p 7860:7860 wildfire-api
```

## 📊 API Usage Examples

### Coordinate-Based Prediction

Predict wildfire risk using geographic coordinates and date:

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "lat": 34.0522,      # Los Angeles latitude
    "lon": -118.2437,    # Los Angeles longitude
    "date": "2024-07-15" # Date for prediction
}

response = requests.post(url, json=data)
result = response.json()

print(f"Fire Occurrence Probability: {result['occurrence_probability']:.2%}")
print(f"Severity Class: {result['severity_class']}")
```

### Direct Weather Data Prediction

Make predictions using your own weather data:

```python
data = {
    "cloud_cover_mean": 25.5,
    "dewpoint_mean": 12.3,
    "evapotranspiration_total": 3.2,
    "fire_weather_index": 8.5,
    "humidity_min": 35.0,
    "pressure_mean": 1013.25,
    "solar_radiation_mean": 200.5,
    "temp_mean": 28.7,
    "temp_range": 12.4,
    "wind_direction_mean": 180.0,
    "wind_direction_std": 15.2,
    "wind_speed_max": 12.8,
    "daynight_N": 0,
    "lat": 34.0522,
    "lon": -118.2437,
    "month": 7
}

response = requests.post("http://localhost:8000/predict/direct", json=data)
```

### Health Check

Monitor API status:

```python
response = requests.get("http://localhost:8000/health")
print(response.json())
# {"status": "healthy", "timestamp": "2024-01-15T10:30:00"}
```

## 🧠 Machine Learning Models

### Model Architecture

The API employs two specialized neural network models:

**Occurrence Model** (Binary Classification)
- **Input**: 23 engineered features
- **Architecture**: Dense neural network with dropout regularization
- **Output**: Probability of wildfire occurrence (0-1)
- **Preprocessing**: StandardScaler normalization

**Severity Model** (Multi-class Classification)
- **Input**: Same 23 engineered features
- **Architecture**: Dense neural network optimized for multi-class output
- **Output**: Probability distribution across 3 severity levels
- **Classes**: Low (0), Moderate (1), High (2)

### Feature Engineering

The API performs sophisticated feature engineering:

```python
# Interaction Features
temp_humidity_index = temperature * (100 - humidity)
temp_dewpoint_diff = temperature - dewpoint
fwi_wind_ratio = fire_weather_index / (wind_speed + 0.1)
radiation_evapo_product = solar_radiation * evapotranspiration

# Cyclical Encoding
lat_cos, lat_sin = cos(lat), sin(lat)
lon_cos, lon_sin = cos(lon), sin(lon)  
month_cos, month_sin = cos(2π * month/12), sin(2π * month/12)

# Fire Weather Index
fwi = (temperature * (100 - humidity)) / (precipitation + 1)
```

### Model Performance Indicators

The models are designed to handle:
- **Temporal patterns**: Seasonal fire behavior through cyclical encoding
- **Geographic patterns**: Location-specific risk through coordinate encoding
- **Weather interactions**: Complex meteorological relationships
- **Real-time prediction**: Low-latency inference suitable for production

## 🌐 API Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/` | GET | API welcome message | None |
| `/predict` | POST | Coordinate-based prediction | `lat`, `lon`, `date` |
| `/predict/direct` | POST | Direct weather data prediction | Weather features object |
| `/health` | GET | API health status | None |
| `/model/info` | GET | Model specifications | None |
| `/date-info` | GET | Date parsing utility | `date` string |

### Response Schema

All prediction endpoints return:

```json
{
  "occurrence_probability": 0.75,
  "occurrence_class": 1,
  "severity_probabilities": [0.1, 0.3, 0.6],
  "severity_class": 2
}
```

## 📁 Project Structure

```
wildfire-prediction-api/
├── main.py                           # FastAPI application
├── Dockerfile                        # Container configuration
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── wildfire_occurrence_model.h5      # Binary classification model
├── wildfire_severity_model.h5        # Multi-class severity model
├── scaler_occured.joblib             # Occurrence model scaler
├── scaler_severity.joblib            # Severity model scaler
└── .gitignore                        # Git ignore rules
```

## 🔧 Technical Implementation

### Key Technologies

- **FastAPI**: High-performance web framework with automatic API documentation
- **TensorFlow/Keras**: Deep learning framework for model inference
- **scikit-learn**: Machine learning utilities and preprocessing
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server for production deployment

### Error Handling & Validation

- **Input Validation**: Pydantic models ensure data type safety
- **Date Validation**: Prevents predictions beyond 14-day forecast limit
- **Weather API Integration**: Robust error handling for external API failures
- **Model Loading**: Graceful handling of missing model files
- **Logging**: Comprehensive logging for debugging and monitoring

### Performance Considerations

- **Model Caching**: Models loaded once at startup
- **Efficient Preprocessing**: Vectorized operations using NumPy
- **Async Architecture**: FastAPI's async capabilities for concurrent requests
- **Memory Management**: Optimized feature engineering pipeline
- **API Timeouts**: Configured timeouts for external weather API calls

## 🚀 Deployment Options

### Local Development
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Container
```bash
docker build -t wildfire-api .
docker run -p 7860:7860 wildfire-api
```

### Cloud Deployment
The API is configured for deployment on:
- **Hugging Face Spaces** (Docker)
- **Google Cloud Run**
- **AWS ECS**
- **Azure Container Instances**

## 📈 Future Enhancements

- [ ] **Batch Prediction Endpoint**: Process multiple locations simultaneously
- [ ] **Historical Analysis**: Time series analysis of fire patterns
- [ ] **Satellite Data Integration**: Incorporate NDVI and thermal imagery
- [ ] **Model Versioning**: A/B testing framework for model updates
- [ ] **Caching Layer**: Redis caching for weather data
- [ ] **Authentication**: API key management and rate limiting
- [ ] **Monitoring Dashboard**: Real-time prediction analytics
- [ ] **Mobile App**: Flutter/React Native mobile interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Your Name** - vijayaragul2005@gmail.com

Project Link: [https://github.com/yourusername/wildfire-prediction-api](https://wildfire-disaster-management-system.onrender.com/docs)

## 🙏 Acknowledgments

- **Open-Meteo API** for providing free weather data
- **TensorFlow team** for the machine learning framework
- **FastAPI community** for the excellent web framework
- **Wildfire research community** for domain expertise and validation

---

*This project demonstrates advanced machine learning engineering practices, including feature engineering, model deployment, API development, and production-ready containerization. It showcases the ability to build end-to-end ML solutions that can be deployed at scale.*
