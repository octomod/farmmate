# main.py - FastAPI Version for Railway Deployment
# Rice Disease Detection API with FastAPI

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Rice Disease Detection API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disease classes
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

DISEASE_INFO = {
    "Bacterial Leaf Blight": {
        "severity": "High",
        "treatment": "Apply copper-based bactericides. Remove infected plants. Use resistant varieties.",
        "prevention": "Use certified seeds, maintain proper drainage, avoid excessive nitrogen."
    },
    "Brown Spot": {
        "severity": "Medium",
        "treatment": "Apply fungicides like Mancozeb or Propiconazole. Improve soil fertility.",
        "prevention": "Use disease-free seeds, apply balanced fertilizers, ensure proper water management."
    },
    "Leaf Blast": {
        "severity": "High",
        "treatment": "Apply Tricyclazole or Isoprothiolane. Remove severely infected leaves.",
        "prevention": "Use resistant varieties, avoid excessive nitrogen, maintain proper spacing."
    },
    "Leaf Scald": {
        "severity": "Medium",
        "treatment": "Apply Propiconazole or Azoxystrobin. Improve field drainage.",
        "prevention": "Use resistant varieties, avoid water stress, maintain field hygiene."
    },
    "Narrow Brown Leaf Spot": {
        "severity": "Low",
        "treatment": "Apply Mancozeb or Copper oxychloride. Improve nutrient balance.",
        "prevention": "Use certified seeds, maintain soil fertility, avoid water stress."
    },
    "Neck Blast": {
        "severity": "Very High",
        "treatment": "Emergency fungicide application (Tricyclazole). Remove infected panicles immediately.",
        "prevention": "Use resistant varieties, monitor closely during heading stage."
    },
    "Rice Hispa": {
        "severity": "Medium",
        "treatment": "Apply insecticides like Chlorpyrifos. Manual removal of adults and larvae.",
        "prevention": "Avoid over-crowding, remove weeds, use resistant varieties."
    },
    "Sheath Blight": {
        "severity": "High",
        "treatment": "Apply Validamycin or Hexaconazole. Improve air circulation.",
        "prevention": "Avoid excessive nitrogen, maintain proper plant spacing, use resistant varieties."
    }
}

class TFLiteModel:
    """Wrapper for TensorFlow Lite model"""
    
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model: {model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logger.info(f"Model loaded - Input: {self.input_details[0]['shape']}, Output: {self.output_details[0]['shape']}")
    
    def predict(self, input_data):
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        return self.interpreter.get_tensor(self.output_details[0]['index'])[0]

# Initialize models
logger.info("Initializing models...")
try:
    binary_model = TFLiteModel('models/rice_health_binary.tflite')
    disease_model = TFLiteModel('models/rice_disease_classifier.tflite')
    logger.info("✅ Models loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")
    raise

def preprocess_image(image_bytes: bytes):
    """Preprocess image for model"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        logger.info(f"Original: {img.size}, {img.mode}")
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((224, 224), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        logger.info(f"Preprocessed: {img_array.shape}, range [{img_array.min():.3f}, {img_array.max():.3f}]")
        
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "Rice Disease Detection API",
        "version": "1.0.0",
        "framework": "FastAPI"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Predict rice disease from uploaded image
    """
    try:
        logger.info(f"Received file: {image.filename}, content_type: {image.content_type}")
        
        # Read image bytes
        image_bytes = await image.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")
        
        # Preprocess
        input_data = preprocess_image(image_bytes)
        
        # Stage 1: Binary classification
        logger.info("Running binary model...")
        binary_output = binary_model.predict(input_data)
        logger.info(f"Binary output: {binary_output}")
        
        diseased_prob = float(binary_output[0]) if len(binary_output) == 1 else float(binary_output[1])
        is_healthy = diseased_prob < 0.5
        binary_confidence = (1 - diseased_prob) * 100 if is_healthy else diseased_prob * 100
        
        logger.info(f"Binary: {'HEALTHY' if is_healthy else 'DISEASED'} ({binary_confidence:.2f}%)")
        
        # Stage 2: Disease classification
        logger.info("Running disease model...")
        disease_output = disease_model.predict(input_data)
        logger.info(f"Disease output: {disease_output}")
        
        max_idx = int(np.argmax(disease_output))
        detected_disease = DISEASE_CLASSES[max_idx]
        disease_confidence = float(disease_output[max_idx]) * 100
        
        logger.info(f"Detected: {detected_disease} ({disease_confidence:.2f}%)")
        
        # Prepare response
        all_diseases = [
            {
                "name": DISEASE_CLASSES[i],
                "probability": round(float(disease_output[i]) * 100, 2)
            }
            for i in range(len(DISEASE_CLASSES))
        ]
        all_diseases.sort(key=lambda x: x['probability'], reverse=True)
        
        result = {
            "success": True,
            "isHealthy": is_healthy,
            "binaryConfidence": round(binary_confidence, 2),
            "diseasedProbability": round(diseased_prob, 4),
            "detectedDisease": detected_disease,
            "diseaseConfidence": round(disease_confidence, 2),
            "allDiseases": all_diseases,
            "diseaseInfo": DISEASE_INFO.get(detected_disease, {})
        }
        
        logger.info("✅ Prediction successful")
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test():
    return {
        "message": "API is working!",
        "endpoints": {
            "POST /predict": "Upload image for disease detection",
            "GET /health": "Health check",
            "GET /": "Service info"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*50)
    logger.info("🌾 Rice Disease Detection API (FastAPI)")
    logger.info("="*50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)