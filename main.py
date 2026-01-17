from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ===============================
# APP CONFIG
# ===============================
app = FastAPI(
    title="FarmMate AI API",
    description="Sri Lanka Agriculture AI System",
    version="1.0"
)

# ===============================
# LOAD MODELS
# ===============================
yield_model = joblib.load("yield_model.pkl")
pest_model = joblib.load("pest_model.pkl")

# ===============================
# LOAD ENCODERS
# ===============================
district_enc = joblib.load("district_encoder.pkl")
season_enc = joblib.load("season_encoder.pkl")
crop_enc = joblib.load("crop_encoder.pkl")
soil_enc = joblib.load("soil_encoder.pkl")

# ===============================
# REQUEST SCHEMAS
# ===============================
class YieldRequest(BaseModel):
    district: str
    season: str
    crop: str
    rainfall_mm: float
    temperature_c: float
    soil_type: str
    fertilizer_kg: float

class PestRequest(BaseModel):
    district: str
    season: str
    crop: str
    temperature_c: float
    rainfall_mm: float
    humidity: float

# ===============================
# RESPONSE SCHEMAS
# ===============================
class YieldResponse(BaseModel):
    predicted_yield_kg_per_ha: float

class PestResponse(BaseModel):
    pest_risk: str

# ===============================
# ROOT ENDPOINT
# ===============================
@app.get("/")
def root():
    return {"message": "FarmMate API running"}

# ===============================
# YIELD PREDICTION
# ===============================
@app.post("/predict-yield", response_model=YieldResponse)
def predict_yield(data: YieldRequest):

    district = district_enc.transform([data.district])[0]
    season = season_enc.transform([data.season])[0]
    crop = crop_enc.transform([data.crop])[0]
    soil = soil_enc.transform([data.soil_type])[0]

    X = pd.DataFrame([[
        district,
        season,
        crop,
        data.rainfall_mm,
        data.temperature_c,
        soil,
        data.fertilizer_kg
    ]], columns=[
        "district",
        "season",
        "crop",
        "rainfall_mm",
        "temperature_c",
        "soil_type",
        "fertilizer_kg"
    ])

    prediction = yield_model.predict(X)[0]

    return YieldResponse(
        predicted_yield_kg_per_ha=round(float(prediction), 2)
    )

# ===============================
# PEST RISK PREDICTION
# ===============================
@app.post("/predict-pest", response_model=PestResponse)
def predict_pest(data: PestRequest):

    district = district_enc.transform([data.district])[0]
    season = season_enc.transform([data.season])[0]
    crop = crop_enc.transform([data.crop])[0]

    X = pd.DataFrame([[
        district,
        season,
        crop,
        data.temperature_c,
        data.rainfall_mm,
        data.humidity
    ]], columns=[
        "district",
        "season",
        "crop",
        "temperature_c",
        "rainfall_mm",
        "humidity"
    ])

    result = pest_model.predict(X)[0]

    return PestResponse(
        pest_risk="High" if int(result) == 1 else "Low"
    )
