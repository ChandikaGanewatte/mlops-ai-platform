from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("models/model.pkl")


# -------------------------
# Input Schema (FIX HERE)
# -------------------------
class InputData(BaseModel):
    Sales: float
    Discount: float
    Year: int
    Month: int
    WeekDay: int


# -------------------------
# API Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "MLOps AI Platform Running"}


@app.post("/predict")
def predict(data: InputData):

    features = np.array([
        data.Sales,
        data.Discount,
        data.Year,
        data.Month,
        data.WeekDay
    ]).reshape(1, -1)

    prediction = model.predict(features)

    return {
        "predicted_profit": float(prediction[0])
    }