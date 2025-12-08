from flask_cors import CORS
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.logger import Logger
from models import predict_task1, predict_task2
from flask import Flask, request, jsonify


logger = Logger()

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({
        "message": "SemEval 2026 API - Clarity & Evasion Classification",
        "usage": "POST /predict with {question: ..., answer: ...}"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "question" not in data or "answer" not in data:
        return jsonify({"error": "Send JSON with 'question' and 'answer'"}), 400

    question = data["question"]
    answer = data["answer"]

    try:
        clarity = predict_task1(question, answer)
        evasion = predict_task2(question, answer)

        logger.log(
            f"[Predicted] Clarity: {clarity} | Evasion: {evasion}", "success")

        return jsonify({
            "question": question,
            "answer": answer,
            "predictions": {
                "clarity": clarity,
                "evasion": evasion
            }
        })

    except Exception as e:
        logger.log(f"Prediction Error: {str(e)}", "error")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.log("Starting SemEval 2026 Flask Server...", "announce")
    app.run(host="0.0.0.0", port=5000, debug=False)
