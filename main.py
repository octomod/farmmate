from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# ======================
# LOAD TRAINED MODELS
# ======================
yield_model = joblib.load("yield_model.pkl")
pest_model = joblib.load("pest_model.pkl")
market_model = joblib.load("market_model.pkl")

# ======================
# HOME TEST ROUTE
# ======================
@app.route("/")
def home():
    return {"status": "FarmMate API is running 🚜"}

# =================================================
# 🌾 RICE YIELD PREDICTION API
# INPUT MATCHES:
# district, season, crop, rainfall_mm, temperature_c,
# soil_type, fertilizer_kg
# =================================================
@app.route("/predict/yield", methods=["POST"])
def predict_yield():
    data = request.json

    input_data = np.array([[
        int(data["district"]),
        int(data["season"]),
        int(data["crop"]),
        float(data["rainfall_mm"]),
        float(data["temperature_c"]),
        int(data["soil_type"]),
        float(data["fertilizer_kg"])
    ]])

    prediction = yield_model.predict(input_data)[0]

    return jsonify({
        "predicted_yield_kg_ha": round(prediction, 2),
        "yield_category": (
            "High" if prediction > 5000 else
            "Medium" if prediction > 3500 else
            "Low"
        )
    })

# =================================================
# 🐛 PEST OUTBREAK PREDICTION API
# INPUT MATCHES:
# district, season, crop, temperature_c, rainfall_mm, humidity
# =================================================
@app.route("/predict/pest", methods=["POST"])
def predict_pest():
    data = request.json

    input_data = np.array([[
        int(data["district"]),
        int(data["season"]),
        int(data["crop"]),
        float(data["temperature_c"]),
        float(data["rainfall_mm"]),
        float(data["humidity"])
    ]])

    prediction = pest_model.predict(input_data)[0]
    probability = pest_model.predict_proba(input_data)[0][1]

    return jsonify({
        "pest_outbreak": bool(prediction),
        "risk_probability": round(probability * 100, 2),
        "alert": "High Risk ⚠️" if probability > 0.7 else "Low Risk ✅"
    })

# =================================================
# 📈 MARKET PRICE FORECAST API
# INPUT MATCHES:
# day, min_price, max_price, volume
# =================================================
@app.route("/predict/price", methods=["POST"])
def predict_price():
    data = request.json

    input_data = np.array([[
        int(data["day"]),
        float(data["min_price"]),
        float(data["max_price"]),
        int(data["volume"])
    ]])

    prediction = market_model.predict(input_data)[0]

    return jsonify({
        "predicted_avg_price": round(prediction, 2),
        "recommendation": (
            "Sell Now 📦" if prediction > data["max_price"]
            else "Hold ⏳"
        )
    })

# ======================
# RAILWAY ENTRY POINT
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
