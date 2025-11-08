# src/models/predict.py
import joblib
import pandas as pd
import os
import json
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_model(model_path="outputs/model.pkl"):
    """Load trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)

def predict_dataframe(df, model_path="outputs/model.pkl"):
    """Run predictions on a DataFrame."""
    model = load_model(model_path)

    # Handle missing columns
    expected_features = model.named_steps["preprocessor"].get_feature_names_out()
    logger.info(f"Predicting on {len(df)} samples...")

    # Run predictions
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    results = [
        {"probability": round(float(p), 2), "label": int(l)}
        for p, l in zip(probs, preds)
    ]
    return results

if __name__ == "__main__":
    df = pd.read_csv("data/processed/processed.csv").drop(columns=["readmitted"])
    results = predict_dataframe(df)
    print(json.dumps(results[:5], indent=4))
