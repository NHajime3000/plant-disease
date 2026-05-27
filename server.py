from flask import Flask, request, jsonify
import requests
import csv
import os
from datetime import datetime

app = Flask(__name__)

CSV_FILE = "edge_ai_results.csv"


def generate_treatment(prediction):

    prompt = f"""
You are a plant disease expert.

Disease detected:
{prediction}

Please provide:

1. Disease description
2. Possible causes
3. Recommended treatment
4. Prevention methods

Keep the answer concise and practical.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:3b",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    return result["response"]


@app.route("/upload", methods=["POST"])
def upload():

    data = request.get_json()

    if not data:
        return jsonify({
            "status": "failed",
            "message": "no json"
        }), 400

    prediction = data.get("prediction", "unknown")

    treatment = generate_treatment(prediction)

    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=[
            "timestamp",
            "model",
            "prediction",
            "confidence",
            "inference_time_ms",
            "fps",
            "model_size_mb",
            "memory_mb",
            "battery",
            "estimated_energy_mj"
        ])

        if not file_exists:
            writer.writeheader()

        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        writer.writerow(data)

    print("\n========================")
    print("Prediction:", prediction)
    print("========================")
    print(treatment)
    print("========================\n")

    return jsonify({
        "status": "success",
        "treatment": treatment
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)