from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

IMG_SIZE = 224

# Load TFLite models
health_interpreter = tf.lite.Interpreter(model_path="rice_health_binary.tflite")
health_interpreter.allocate_tensors()

disease_interpreter = tf.lite.Interpreter(model_path="rice_disease_classifier.tflite")
disease_interpreter.allocate_tensors()

health_input = health_interpreter.get_input_details()
health_output = health_interpreter.get_output_details()

disease_input = disease_interpreter.get_input_details()
disease_output = disease_interpreter.get_output_details()

DISEASE_CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Leaf Spot",
    "Neck Blast",
    "Rice Hispa",
    "Sheath Blight"
]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    return np.expand_dims(image.astype(np.float32), axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)

    # STEP 1: Healthy vs Diseased
    health_interpreter.set_tensor(health_input[0]['index'], img)
    health_interpreter.invoke()
    health_pred = health_interpreter.get_tensor(health_output[0]['index'])[0][0]

    if health_pred < 0.5:
        return {
            "status": "Healthy",
            "confidence": round((1 - health_pred) * 100, 2)
        }

    # STEP 2: Disease classification
    disease_interpreter.set_tensor(disease_input[0]['index'], img)
    disease_interpreter.invoke()
    disease_preds = disease_interpreter.get_tensor(disease_output[0]['index'])[0]

    index = np.argmax(disease_preds)
    confidence = float(disease_preds[index]) * 100

    return {
        "status": "Diseased",
        "disease": DISEASE_CLASSES[index],
        "confidence": round(confidence, 2)
    }
