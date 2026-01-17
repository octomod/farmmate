from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# ===============================
# Load datasets ONCE
# ===============================
yield_df = pd.read_csv("sri_lanka_yield_data.csv")
pest_df = pd.read_csv("sri_lanka_pest_data.csv")
market_df = pd.read_csv("sri_lanka_market_prices.csv")

# ===============================
# Utility functions
# ===============================
def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

# ===============================
# Health Check
# ===============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "FarmMate API running",
        "services": [
            "Yield Prediction",
            "Pest Warning",
            "Market Prices"
        ]
    })

# ===============================
# Yield Prediction
# ===============================
@app.route("/predict/yield", methods=["POST"])
def predict_yield():
    try:
        data = request.get_json(force=True)

        district = data["district"]
        season = data["season"]
        crop = data["crop"]
        rainfall = safe_float(data["rainfall_mm"])
        temperature = safe_float(data["temperature_c"])
        fertilizer = safe_float(data["fertilizer_kg"])

        filtered = yield_df[
            (yield_df["district"] == district) &
            (yield_df["season"] == season) &
            (yield_df["crop"] == crop)
        ]

        if filtered.empty:
            return jsonify({
                "success": False,
                "message": "No yield data found"
            })

        avg_yield = filtered["yield_kg_ha"].mean()

        # Simple real-world adjustment logic
        predicted_yield = avg_yield + (
            (rainfall - filtered["rainfall_mm"].mean()) * 1.5 +
            (temperature - filtered["temperature_c"].mean()) * 10 +
            (fertilizer - filtered["fertilizer_kg"].mean()) * 0.8
        )

        return jsonify({
            "success": True,
            "predicted_yield_kg_ha": round(predicted_yield, 2)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ===============================
# Pest Warning
# ===============================
@app.route("/predict/pest", methods=["POST"])
def predict_pest():
    try:
        data = request.get_json(force=True)

        district = data["district"]
        season = data["season"]
        crop = data["crop"]
        temperature = safe_float(data["temperature_c"])
        rainfall = safe_float(data["rainfall_mm"])
        humidity = safe_float(data["humidity"])

        filtered = pest_df[
            (pest_df["district"] == district) &
            (pest_df["season"] == season) &
            (pest_df["crop"] == crop)
        ]

        if filtered.empty:
            return jsonify({
                "success": False,
                "message": "No pest data found"
            })

        avg_pest = filtered["pest_level"].mean()

        risk_score = (
            (temperature / 40) +
            (rainfall / 300) +
            (humidity / 100) +
            avg_pest
        ) / 4

        if risk_score > 0.65:
            level = "High"
        elif risk_score > 0.35:
            level = "Medium"
        else:
            level = "Low"

        return jsonify({
            "success": True,
            "pest_risk_level": level
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ===============================
# Market Price (Latest)
# ===============================
@app.route("/predict/market", methods=["POST"])
def predict_market():
    try:
        data = request.get_json(force=True)

        crop = data["crop"]
        market = data["market"]

        filtered = market_df[
            (market_df["crop"] == crop) &
            (market_df["market"] == market)
        ]

        if filtered.empty:
            return jsonify({
                "success": False,
                "message": "No market data found"
            })

        latest = filtered.iloc[-1]

        return jsonify({
            "success": True,
            "crop": crop,
            "market": market,
            "min_price": float(latest["min_price"]),
            "max_price": float(latest["max_price"]),
            "avg_price": float(latest["avg_price"]),
            "volume": int(latest["volume"])
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
