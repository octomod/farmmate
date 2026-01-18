from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# LOAD MODELS (ONCE)
# -------------------------------
health_model = tf.lite.Interpreter(model_path="rice_health_binary.tflite")
health_model.allocate_tensors()

disease_model = tf.lite.Interpreter(model_path="rice_disease_classifier.tflite")
disease_model.allocate_tensors()

health_input = health_model.get_input_details()
health_output = health_model.get_output_details()

disease_input = disease_model.get_input_details()
disease_output = disease_model.get_output_details()

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

# -------------------------------
# IMAGE PREPROCESS
# -------------------------------
def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# -------------------------------
# PREDICT ENDPOINT
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = preprocess(image_bytes)

        # ----- HEALTH CHECK -----
        health_model.set_tensor(health_input[0]["index"], img)
        health_model.invoke()
        health_prob = health_model.get_tensor(
            health_output[0]["index"]
        )[0][0]

        # Healthy
        if health_prob < 0.5:
            return {
                "status": "Healthy",
                "confidence": round((1 - health_prob) * 100, 2)
            }

        # ----- DISEASE CLASSIFIER -----
        disease_model.set_tensor(disease_input[0]["index"], img)
        disease_model.invoke()
        preds = disease_model.get_tensor(
            disease_output[0]["index"]
        )[0]

        index = int(np.argmax(preds))
        confidence = float(preds[index] * 100)

        return {
            "status": "Diseased",
            "disease": DISEASE_CLASSES[index],
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }
