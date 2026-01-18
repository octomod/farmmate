from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# LOAD MODELS (ON START)
# ==========================
try:
    health_model = tf.lite.Interpreter(model_path="rice_health_binary.tflite")
    health_model.allocate_tensors()

    disease_model = tf.lite.Interpreter(model_path="rice_disease_classifier.tflite")
    disease_model.allocate_tensors()

except Exception as e:
    print("MODEL LOAD ERROR:", e)
    raise e

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
# IMAGE PREPROCESS
# ==========================
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# ==========================
# PREDICT ENDPOINT
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file:
            return {"error": "No image received"}

        image_bytes = await file.read()
        input_data = preprocess_image(image_bytes)

        # ---- HEALTH MODEL ----
        health_input = health_model.get_input_details()
        health_output = health_model.get_output_details()
        health_model.set_tensor(health_input[0]["index"], input_data)
        health_model.invoke()
        health_pred = health_model.get_tensor(health_output[0]["index"])[0][0]

        if health_pred < 0.5:
            return {
                "status": "Healthy",
                "confidence": round((1 - health_pred) * 100, 2)
            }

        # ---- DISEASE MODEL ----
        disease_input = disease_model.get_input_details()
        disease_output = disease_model.get_output_details()
        disease_model.set_tensor(disease_input[0]["index"], input_data)
        disease_model.invoke()
        preds = disease_model.get_tensor(disease_output[0]["index"])[0]

        idx = int(np.argmax(preds))
        confidence = float(preds[idx]) * 100

        return {
            "status": "Diseased",
            "disease": DISEASE_CLASSES[idx],
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "error": "Prediction failed",
            "details": str(e)
        }
