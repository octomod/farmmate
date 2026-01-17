from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# ===============================
# Load ML models ONCE
# ===============================
yield_model = joblib.load("yield_model.pkl")
pest_model = joblib.load("pest_model.pkl")
market_model = joblib.load("market_model.pkl")

# ===============================
# Helpers
# ===============================
def to_float(x):
    try:
        return float(x)
    except:
        return 0.0

# ===============================
# Health
# ===============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "FarmMate ML API running",
        "models_loaded": True
    })

# ===============================
# Yield Prediction
# ===============================
@app.route("/predict/yield", methods=["POST"])
def predict_yield():
    try:
        data = request.get_json(force=True)

        X = np.array([[
            to_float(data["rainfall_mm"]),
            to_float(data["temperature_c"]),
            to_float(data["fertilizer_kg"])
        ]])

        prediction = yield_model.predict(X)[0]

        return jsonify({
            "success": True,
            "predicted_yield_kg_ha": round(float(prediction), 2)
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

        X = np.array([[
            to_float(data["temperature_c"]),
            to_float(data["rainfall_mm"]),
            to_float(data["humidity"])
        ]])

        risk = pest_model.predict(X)[0]

        levels = {0: "Low", 1: "Medium", 2: "High"}

        return jsonify({
            "success": True,
            "pest_risk_level": levels.get(int(risk), "Unknown")
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ===============================
# Market Price Prediction
# ===============================
@app.route("/predict/market", methods=["POST"])
def predict_market():
    try:
        data = request.get_json(force=True)

        X = np.array([[
            to_float(data["min_price"]),
            to_float(data["max_price"]),
            to_float(data["volume"])
        ]])

        predicted_price = market_model.predict(X)[0]

        return jsonify({
            "success": True,
            "predicted_avg_price": round(float(predicted_price), 2)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ===============================
# Run
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
