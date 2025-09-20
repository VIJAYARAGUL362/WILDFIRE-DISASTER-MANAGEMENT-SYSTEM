# Use a specific Python 3.10-slim base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the other files, including your Python script and models
COPY . .

# Expose the port your FastAPI app will listen on
EXPOSE 7860

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "wildfire_api:app", "--host", "0.0.0.0", "--port", "7860"]