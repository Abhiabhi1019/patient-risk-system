from flask import Flask, request, jsonify
from src.models.predict import load_artifacts, predict_dataframe
from src.utils.logger import get_logger
import pandas as pd

logger = get_logger("api")
app = Flask(__name__)

# ✅ Load model, preprocessor, and expected schema once at startup
model, preprocessor, expected_raw_columns = load_artifacts()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        # Ensure input is a list of records (JSON)
        if not isinstance(input_data, list):
            input_data = [input_data]

        df = pd.DataFrame(input_data)
        logger.info(f"Received data for prediction: {df.shape}")

        # ✅ Pass expected_raw_columns to predict_dataframe
        preds = predict_dataframe(df, model, preprocessor, expected_raw_columns)

        return jsonify(preds)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # ✅ Flask app runs on port 8000 for Docker mapping
    app.run(host="0.0.0.0", port=8000)

