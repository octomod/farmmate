# main.py - Railway-optimized FastAPI backend
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import logging
import os

# Try lightweight TFLite Runtime first
try:
    import tflite_runtime.interpreter as tflite
    USE_TFLITE_RUNTIME = True
    logging.info("Using TFLite Runtime (lightweight)")
except ImportError:
    import tensorflow as tf
    USE_TFLITE_RUNTIME = False
    logging.info("Using full TensorFlow")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Rice Disease Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DISEASE_CLASSES = [
    "Bacterial Leaf Blight", "Brown Spot", "Leaf Blast", "Leaf Scald",
    "Narrow Brown Leaf Spot", "Neck Blast", "Rice Hispa", "Sheath Blight"
]

class TFLiteModel:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading: {model_path}")
        
        if USE_TFLITE_RUNTIME:
            self.interpreter = tflite.Interpreter(model_path=model_path)
        else:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
        
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        logger.info(f"✅ Loaded: {model_path}")
    
    def predict(self, input_data):
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])[0]

# Load models
logger.info("🔧 Initializing models...")
try:
    binary_model = TFLiteModel('models/rice_health_binary.tflite')
    disease_model = TFLiteModel('models/rice_disease_classifier.tflite')
    logger.info("✅ All models loaded")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    binary_model = None
    disease_model = None

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.get("/")
async def root():
    return {"status": "healthy", "service": "Rice Disease Detection"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": binary_model is not None,
        "runtime": "tflite-runtime" if USE_TFLITE_RUNTIME else "tensorflow"
    }

@app.get("/debug")
async def debug():
    return {
        "models_folder": os.path.exists("models"),
        "binary_model": os.path.exists("models/rice_health_binary.tflite"),
        "disease_model": os.path.exists("models/rice_disease_classifier.tflite"),
        "files": os.listdir("models") if os.path.exists("models") else []
    }

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        if binary_model is None or disease_model is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        logger.info(f"📥 Received: {image.filename}")
        
        image_bytes = await image.read()
        input_data = preprocess_image(image_bytes)
        
        # Binary classification
        binary_output = binary_model.predict(input_data)
        diseased_prob = float(binary_output[0] if len(binary_output) == 1 else binary_output[1])
        is_healthy = diseased_prob < 0.5
        
        # Disease classification
        disease_output = disease_model.predict(input_data)
        max_idx = int(np.argmax(disease_output))
        
        result = {
            "success": True,
            "isHealthy": is_healthy,
            "binaryConfidence": round((1 - diseased_prob) * 100 if is_healthy else diseased_prob * 100, 2),
            "detectedDisease": DISEASE_CLASSES[max_idx],
            "diseaseConfidence": round(float(disease_output[max_idx]) * 100, 2),
            "allDiseases": sorted([
                {"name": DISEASE_CLASSES[i], "probability": round(float(disease_output[i]) * 100, 2)}
                for i in range(len(DISEASE_CLASSES))
            ], key=lambda x: x['probability'], reverse=True)
        }
        
        logger.info(f"✅ Success: {result['detectedDisease']}")
        return result
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvic