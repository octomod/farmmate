# main.py - Rice Disease Detection API
# Professional Flask backend for TFLite model inference

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# Disease classes - MUST match your training data order
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

# Disease information
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
    """Wrapper class for TensorFlow Lite model inference"""
    
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Input shape: {self.input_details[0]['shape']}")
        logger.info(f"Output shape: {self.output_details[0]['shape']}")
    
    def predict(self, input_data):
        """Run inference on input data"""
        # Ensure input data has correct shape
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output_data[0]

# Initialize models
logger.info("Initializing models...")
try:
    binary_model = TFLiteModel('rice_health_binary.tflite')
    disease_model = TFLiteModel('rice_disease_classifier.tflite')
    logger.info("✅ All models loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load models: {str(e)}")
    logger.error("Make sure .tflite files are in 'models/' folder")
    raise

def preprocess_image(image_file):
    """
    Preprocess image for model inference
    Matches Colab training: 224x224, RGB, normalized to [0, 1]
    """
    try:
        # Read image from file upload
        img = Image.open(io.BytesIO(image_file.read()))
        logger.info(f"Original image size: {img.size}, mode: {img.mode}")
        
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.info("Converted image to RGB")
        
        # Resize to 224x224 (same as training)
        img = img.resize((224, 224), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1] (CRITICAL: must match training)
        img_array = img_array / 255.0
        
        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Preprocessed shape: {img_array.shape}")
        logger.info(f"Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
        logger.info(f"Mean: {img_array.mean():.3f}")
        
        return img_array
    
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Rice Disease Detection API",
        "version": "1.0.0",
        "models": {
            "binary": "rice_health_binary.tflite",
            "disease": "rice_disease_classifier.tflite"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Simple health check"""
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Expects: multipart/form-data with 'image' field
    Returns: JSON with disease prediction results
    """
    try:
        # Validate request
        if 'image' not in request.files:
            logger.warning("No image in request")
            return jsonify({"error": "No image uploaded"}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            logger.warning("Empty filename")
            return jsonify({"error": "No image selected"}), 400
        
        logger.info(f"Processing image: {image_file.filename}")
        
        # Preprocess image
        input_data = preprocess_image(image_file)
        
        # STAGE 1: Binary Classification (Healthy vs Diseased)
        logger.info("Running binary classification...")
        binary_output = binary_model.predict(input_data)
        logger.info(f"Binary output: {binary_output}")
        
        # Handle different output formats
        if len(binary_output) == 1:
            # Sigmoid output: single value
            diseased_prob = float(binary_output[0])
        elif len(binary_output) == 2:
            # Softmax output: [healthy_prob, diseased_prob]
            diseased_prob = float(binary_output[1])
        else:
            logger.error(f"Unexpected binary output length: {len(binary_output)}")
            return jsonify({"error": "Invalid model output"}), 500
        
        # Determine if healthy or diseased (threshold: 0.5)
        is_healthy = diseased_prob < 0.5
        binary_confidence = (1 - diseased_prob) * 100 if is_healthy else diseased_prob * 100
        
        logger.info(f"Binary result: {'HEALTHY' if is_healthy else 'DISEASED'} ({binary_confidence:.2f}%)")
        
        # STAGE 2: Disease Classification (8 diseases)
        logger.info("Running disease classification...")
        disease_output = disease_model.predict(input_data)
        logger.info(f"Disease output: {disease_output}")
        
        # Convert to list for JSON serialization
        disease_probs = disease_output.tolist()
        
        # Find disease with highest probability
        max_idx = int(np.argmax(disease_output))
        detected_disease = DISEASE_CLASSES[max_idx]
        disease_confidence = float(disease_output[max_idx]) * 100
        
        logger.info(f"Top disease: {detected_disease} ({disease_confidence:.2f}%)")
        
        # Prepare all disease probabilities
        all_diseases = [
            {
                "name": DISEASE_CLASSES[i],
                "probability": round(float(disease_probs[i]) * 100, 2)
            }
            for i in range(len(DISEASE_CLASSES))
        ]
        
        # Sort by probability (highest first)
        all_diseases_sorted = sorted(all_diseases, key=lambda x: x['probability'], reverse=True)
        
        # Get disease info
        disease_info = DISEASE_INFO.get(detected_disease, {})
        
        # Prepare response
        result = {
            "success": True,
            "isHealthy": is_healthy,
            "binaryConfidence": round(binary_confidence, 2),
            "diseasedProbability": round(diseased_prob, 4),
            "detectedDisease": detected_disease,
            "diseaseConfidence": round(disease_confidence, 2),
            "allDiseases": all_diseases_sorted,
            "diseaseInfo": disease_info
        }
        
        logger.info(f"✅ Prediction successful")
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify API is working"""
    return jsonify({
        "message": "API is working!",
        "endpoints": {
            "POST /predict": "Upload image for disease detection",
            "GET /health": "Health check",
            "GET /": "Service info"
        }
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Check if models exist
    if not os.path.exists('models'):
        logger.error("❌ 'models' folder not found!")
        logger.error("Create a 'models' folder and add your .tflite files")
        exit(1)
    
    if not os.path.exists('models/rice_health_binary.tflite'):
        logger.error("❌ rice_health_binary.tflite not found in models/")
        exit(1)
    
    if not os.path.exists('models/rice_disease_classifier.tflite'):
        logger.error("❌ rice_disease_classifier.tflite not found in models/")
        exit(1)
    
    logger.info("="*50)
    logger.info("🌾 Rice Disease Detection API")
    logger.info("="*50)
    logger.info("Server starting...")
    logger.info("Access at: http://localhost:5000")
    logger.info("Health check: http://localhost:5000/health")
    logger.info("="*50)
    
    # Run server
    # For production, use gunicorn or waitress
    app.run(
        host='0.0.0.0',  # Accessible from network
        port=5000,
        debug=True  # Set to False in production
    )