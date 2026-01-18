from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import os
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

IMG_SIZE = 224

DISEASE_CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight",
    "Narrow Brown Leaf Spot",
    "Neck Blast",
    "Rice Hispa"
]

# ==========================
# MODEL LOAD WITH VERIFICATION
# ==========================
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"MODEL NOT FOUND: {path}")
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

try:
    health_model = load_model("rice_health_binary.tflite")
    disease_model = load_model("rice_disease_classifier.tflite")
    print("✅ MODELS LOADED SUCCESSFULLY")
except Exception as e:
    print("❌ MODEL LOAD FAILED")
    traceback.print_exc()
    raise e

# ==========================
# HEALTH CHECK (CRITICAL)
# ==========================
@app.get("/")
def root():
    return {
        "status": "API RUNNING",
        "models_loaded": True
    }

# ==========================
# IMAGE PREPROCESS
# ==========================
def preprocess_image(bytes_data):
    image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ==========================
# PREDICT
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if file is None:
            return {"error": "NO_FILE"}

        img_bytes = await file.read()
        input_tensor = preprocess_image(img_bytes)

        # HEALTH
        h_in = health_model.get_input_details()[0]["index"]
        h_out = health_model.get_output_details()[0]["index"]
        health_model.set_tensor(h_in, input_tensor)
        health_model.invoke()
        health_score = health_model.get_tensor(h_out)[0][0]

        if health_score < 0.5:
            return {
                "status": "Healthy",
                "confidence": round((1 - health_score) * 100, 2)
            }

        # DISEASE
        d_in = disease_model.get_input_details()[0]["index"]
        d_out = disease_model.get_output_details()[0]["index"]
        disease_model.set_tensor(d_in, input_tensor)
        disease_model.invoke()
        preds = disease_model.get_tensor(d_out)[0]

        idx = int(np.argmax(preds))

        return {
            "status": "Diseased",
            "disease": DISEASE_CLASSES[idx],
            "confidence": round(float(preds[idx]) * 100, 2)
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "error": "PREDICTION_FAILED",
            "details": str(e)
        }
