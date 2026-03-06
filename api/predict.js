import fetch from "node-fetch";

export default async function handler(req, res) {
    if (req.method !== "POST") {
        return res.status(405).json({ error: "Method not allowed" });
    }

    const ROBOFLOW_URL =
        "https://serverless.roboflow.com/tes-elulw/workflows/farmmate-riceleafdiseasedetection";

    const API_KEY = "5QgFyxF5bz2ra2mtl3KS";

    try {
        const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
        const { imageUrl } = body;

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

        if (!data.outputs || !Array.isArray(data.outputs) || data.outputs.length === 0) {
            return res.status(500).json({
                error: "Invalid Roboflow response",
                roboflow: data,
            });
        }

        const predictions = data.outputs[0].predictions;

        return res.status(200).json({
            disease: predictions.top,
            confidence: predictions.confidence,
        });

    } catch (err) {
        return res.status(500).json({ error: err.message });
    }
}