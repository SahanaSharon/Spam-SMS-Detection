import pickle
import os
from flask import Flask, request, render_template, flash, redirect, url_for

app = Flask(__name__)
app.secret_key = "replace_with_a_random_secret"  # for flash messages

MODEL_PATH = os.path.join("model", "model.pkl")

# Load the trained model
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None, f"Model not found at {path}. Train the model first."
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"

model, load_err = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob_spam = None

    if load_err:
        flash(load_err, "danger")

    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if not message:
            flash("Please enter a message to classify.", "warning")
            return redirect(url_for("index"))

        if model is None:
            flash("Model not loaded.", "danger")
            return redirect(url_for("index"))

        try:
            # Predict
            pred = model.predict([message])[0]
            prediction = "SPAM" if pred == "spam" else "HARM"

            # Probability (if available)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([message])[0]
                classes = model.classes_
                try:
                    idx = list(classes).index("spam")
                    prob_spam = round(probs[idx] * 100, 2)
                except ValueError:
                    prob_spam = None

        except Exception as e:
            flash(f"Prediction error: {e}", "danger")

    return render_template("index.html", prediction=prediction, prob_spam=prob_spam)

if __name__ == "__main__":
    app.run(debug=True)
