from flask import Flask, request, jsonify
import numpy as np
import traceback

app = Flask(__name__)

# -------------------------------
# Dummy ML logic (replace later)
# -------------------------------

def predict_yield(features):
    # Example logic (replace with trained model)
    return round(float(np.sum(features) * 1.25), 2)

def predict_pest(features):
    # Example logic (replace with trained model)
    risk_score = np.mean(features)
    if risk_score > 0.6:
        return "High"
    elif risk_score > 0.3:
        return "Medium"
    else:
        return "Low"


# -------------------------------
# Yield Prediction API
# -------------------------------
@app.route("/predict/yield", methods=["POST"])
def yield_prediction():
    try:
        data = request.get_json(force=True)

        features = data.get("features")
        if not features or not isinstance(features, list):
            return jsonify({
                "success": False,
                "error": "Invalid or missing features"
            }), 400

        features = np.array(features, dtype=float)
        result = predict_yield(features)

        return jsonify({
            "success": True,
            "predicted_yield": result,
            "unit": "kg/acre"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -------------------------------
# Pest Warning API
# -------------------------------
@app.route("/predict/pest", methods=["POST"])
def pest_prediction():
    try:
        data = request.get_json(force=True)

        features = data.get("features")
        if not features or not isinstance(features, list):
            return jsonify({
                "success": False,
                "error": "Invalid or missing features"
            }), 400

        features = np.array(features, dtype=float)
        risk = predict_pest(features)

        return jsonify({
            "success": True,
            "pest_risk_level": risk
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -------------------------------
# Health Check
# -------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "API running",
        "services": ["Yield Prediction", "Pest Warning"]
    })


# -------------------------------
# App Runner
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
