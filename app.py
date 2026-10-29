from flask import Flask, request, jsonify
from pycaret.classification import load_model
from datetime import datetime, timezone
import pandas as pd
import logging

app = Flask(__name__)
model = load_model("/home/ubuntu/app/iris_model")

logging.basicConfig(
    filename="/home/ubuntu/app/ml_api.log",
    level=logging.INFO
)

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({
                "error": "Provide features in the JSON request"
            }), 400

        features = data["features"]

        if len(features) != 4:
            return jsonify({
                "error": "Expected exactly four features"
            }), 400

        # Preserve the feature names expected by PyCaret
        input_df = pd.DataFrame(
            [features],
            columns=FEATURES
        )

        prediction = model.predict(input_df)[0]

        logging.info({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": features,
            "prediction": str(prediction)
        })

        return jsonify({
            "prediction": str(prediction)
        })

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
