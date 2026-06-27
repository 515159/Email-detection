from __future__ import annotations

from pathlib import Path

import joblib
from flask import Flask, render_template, request


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "phishing_model.joblib"

app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run 'python train_model.py' first."
        )
    return joblib.load(MODEL_PATH)


model = load_model()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    email_text = ""
    error = None

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()

        if not email_text:
            error = "Please paste an email before checking it."
        else:
            prediction = model.predict([email_text])[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([email_text])[0]
                class_index = list(model.classes_).index(prediction)
                confidence = round(probabilities[class_index] * 100, 1)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        email_text=email_text,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=False)
