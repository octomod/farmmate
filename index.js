import express from "express";
import fetch from "node-fetch";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

const ROBOFLOW_URL =
  "https://serverless.roboflow.com/tes-elulw/workflows/farmmate-riceleafdiseasedetection";

const API_KEY = process.env.ROBOFLOW_API_KEY; // Railway env var

app.post("/predict", async (req, res) => {
  try {
    const { imageUrl } = req.body;

    if (!imageUrl) {
      return res.status(400).json({ error: "imageUrl required" });
    }

    const response = await fetch(ROBOFLOW_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
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

    const data = await response.json();

    const prediction = data.outputs[0].predictions;

    res.json({
      disease: prediction.top,
      confidence: prediction.confidence,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/", (req, res) => {
  res.send("Backend running");
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
