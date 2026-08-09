from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained model
model = joblib.load("model.joblib")

# Create FastAPI application
app = FastAPI(
    title="Boston Housing Price Prediction API",
    description="Predict house prices using a trained ML regression model",
    version="1.0"
)


# Request schema
class HouseFeatures(BaseModel):
    features: list[float]


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to the Boston Housing Price Prediction API!",
        "docs": "/docs"
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: HouseFeatures):

    prediction = model.predict([data.features])[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }
