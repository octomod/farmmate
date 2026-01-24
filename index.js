import express from "express";
import fetch from "node-fetch";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

const ROBOFLOW_URL =
  "https://serverless.roboflow.com/tes-elulw/workflows/farmmate-riceleafdiseasedetection";

const API_KEY = process.env.ROBOFLOW_API_KEY;

app.post("/predict", async (req, res) => {
  try {
    const { imageUrl } = req.body;

    if (!imageUrl) {
      return res.status(400).json({ error: "imageUrl is required" });
    }

    const rfResponse = await fetch(ROBOFLOW_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: API_KEY,
        inputs: {
          image: {
            type: "url",
            value: imageUrl,
          },
        },
      }),
    });

    const data = await rfResponse.json();

    // ✅ SAFETY CHECK (THIS FIXES YOUR ERROR)
    if (!data.outputs || !Array.isArray(data.outputs) || data.outputs.length === 0) {
      return res.status(500).json({
        error: "Invalid Roboflow response",
        roboflow: data,
      });
    }

    const predictions = data.outputs[0].predictions;

    if (!predictions) {
      return res.status(500).json({
        error: "No predictions found",
        roboflow: data,
      });
    }

    return res.json({
      disease: predictions.top,
      confidence: predictions.confidence,
    });

  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

app.get("/", (req, res) => {
  res.send("Backend running");
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log("Server running on port", PORT);
});
