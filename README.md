# Wildfire Prediction API

## Overview

The Wildfire Prediction API is a robust and professional-grade system designed to predict the occurrence and severity of wildfires. Utilizing a combination of machine learning models and real-time weather data, this API provides valuable insights for risk assessment and preparedness. The API is built with FastAPI, ensuring high performance, easy documentation, and scalable architecture.

## Features

  * **AI-Powered Predictions**: The system uses two separate AI models—one for predicting the probability of a wildfire's occurrence and another for classifying its potential severity.
  * **Real-Time Weather Integration**: It automatically fetches hourly and daily weather data (temperature, humidity, wind, solar radiation, etc.) from the Open-Meteo API to generate accurate predictions for any given location and date.
  * **Comprehensive Feature Engineering**: The API constructs a rich set of features, including a custom-calculated Fire Weather Index (FWI), solar radiation, and geographical and temporal factors, to feed the prediction models.
  * **Health Monitoring**: A dedicated `/health` endpoint allows for easy monitoring of the API's status and model readiness.
  * **Scalable and Asynchronous**: Built with FastAPI, the API is asynchronous and ready for high-concurrency environments.
  * **OpenAPI Documentation**: Automatic interactive API documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc), making it easy for developers to integrate.

## Endpoints

### 1\. Root Endpoint

`GET /`

Returns a welcome message and a list of available endpoints.

**Response:**

```json
{
  "message": "Wildfire Prediction API",
  "version": "1.0",
  "endpoints": [
    "/predict",
    "/health",
    "/docs"
  ]
}
```

### 2\. Health Check

`GET /health`

Provides the current health status of the API, including whether the machine learning models have been successfully loaded.

**Response:**

```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### 3\. Prediction Endpoint

`POST /predict`

Accepts geographical coordinates and a date to predict wildfire occurrence and severity.

**Request Body:**

```json
{
  "lat": 34.0522,
  "lon": -118.2437,
  "date": "2025-08-18"
}
```

**Response:**

```json
{
  "occurrence_probability": 0.85,
  "occurrence_class": 1,
  "severity_probabilities": [0.15, 0.70, 0.15],
  "severity_class": 1
}
```

  * `occurrence_probability`: The likelihood of a wildfire occurring (0.0 to 1.0).
  * `occurrence_class`: A binary classification (0 for no fire, 1 for fire) based on a threshold.
  * `severity_probabilities`: A list of probabilities for each severity class.
  * `severity_class`: The predicted severity level (0, 1, or 2), determined by a per-class threshold logic.

### 4\. Documentation

  * **Swagger UI**: `GET /docs`
  * **ReDoc**: `GET /redoc`

## Technology Stack

  * **Web Framework**: FastAPI
  * **Machine Learning**: TensorFlow/Keras and scikit-learn (via `joblib`)
  * **Data Handling**: NumPy
  * **External APIs**: Open-Meteo for weather data
  * **Dependency Management**: `pip` (See `requirements.txt` for details)

## Setup and Installation

### Prerequisites

  * Python 3.8+
  * `pip` for package installation

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Place Models:**
    Ensure you have the trained models (`wildfire_occurrence_model.keras`, `occurence_scaler.joblib`, `wildfire_severity_model.keras`, `severity_scaler.joblib`) in the root directory of the project.

5.  **Run the application:**

    ```bash
    uvicorn main:app --reload
    ```

    The API will be available at `http://127.0.0.1:8000`.

## Model Information

The API relies on two pre-trained neural network models:

  * `wildfire_occurrence_model.keras`: A model trained to predict the binary outcome of wildfire occurrence. It's paired with `occurence_scaler.joblib`, a scaler used to normalize the input features.
  * `wildfire_severity_model.keras`: A multi-class classification model that predicts the severity level. It's paired with `severity_scaler.joblib`.

During startup, the `load_models` function attempts to load these files from the local directory. If the models are not found or an error occurs during loading, the API will run but the `/predict` endpoint will return a `503 Service Unavailable` error, as indicated by the `/health` endpoint's status.

## Error Handling

  * **`400 Bad Request`**: Returned if the request body is malformed, for example, if the `date` format is incorrect.
  * **`500 Internal Server Error`**: Returned for issues with the external weather API or errors during feature generation or prediction.
  * **`503 Service Unavailable`**: Returned if the machine learning models fail to load at startup. The `/health` endpoint can be used to diagnose this issue.
