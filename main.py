from fastapi import FastAPI
from pydantic import BaseModel
from inference_sdk import InferenceHTTPClient

app = FastAPI()

# 🔐 Roboflow client (API key stays here)
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="gHzY8fywgOI88avYaBnD"  # ⚠️ rotate later
)

# 📦 Request body schema
class ImageRequest(BaseModel):
    image_url: str

@app.post("/predict")
def predict_disease(data: ImageRequest):
    try:
        result = client.run_workflow(
            workspace_name="tes-elulw",
            workflow_id="farmmate-riceleafdiseasedetection",
            images={
                "image": data.image_url
            },
            use_cache=True
        )

        # Extract clean result
        prediction = result["outputs"][0]["predictions"]

        return {
            "disease": prediction["top"],
            "confidence": prediction["confidence"]
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "Backend running"}
