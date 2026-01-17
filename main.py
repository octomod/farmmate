from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

yield_model = joblib.load("yield_model.pkl")
pest_model = joblib.load("pest_model.pkl")

@app.post("/predict-yield")
def predict_yield(data: dict):
    features = np.array([[
        data["district"],
        data["season"],
        data["crop"],
        data["rainfall_mm"],
        data["temperature_c"],
        data["soil_type"],
        data["fertilizer_kg"]
    ]])

    prediction = yield_model.predict(features)[0]
    return {
        "predicted_yield": round(float(prediction), 2),
        "unit": "kg/ha"
    }

@app.post("/predict-pest")
def predict_pest(data: dict):
    features = np.array([[
        data["district"],
        data["season"],
        data["crop"],
        data["temperature_c"],
        data["rainfall_mm"],
        data["humidity"]
    ]])

    result = pest_model.predict(features)[0]
    return {
        "pest_risk": "High" if result == 1 else "Low"
    }
