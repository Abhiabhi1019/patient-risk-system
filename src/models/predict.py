import os
import json
import joblib
import pandas as pd

def load_artifacts(model_dir="outputs"):
    model_path = os.path.join(model_dir, "model.joblib")
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    schema_path = os.path.join(model_dir, "feature_schema.json")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    expected_raw_columns = []
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            schema = json.load(f)
            expected_raw_columns = schema.get("raw_columns", [])

    print(f"[INFO] ✅ Loaded model and preprocessor from {model_dir}")
    return model, preprocessor, expected_raw_columns


def predict_dataframe(df, model, preprocessor=None, expected_raw_columns=None):
    """
    Run predictions on a given dataframe using the trained model pipeline.
    Handles both pipeline-with-preprocessor and separate model+preprocessor setups.
    """

    import numpy as np

    # 🧩 Ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=expected_raw_columns or [])

    df.columns = df.columns.map(str)

    if expected_raw_columns:
        for col in expected_raw_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_raw_columns]

    # ✅ Step 1 — Try predicting directly with model
    try:
        proba = model.predict_proba(df)[:, 1]
    except Exception:
        # If model is not a full pipeline, use the preprocessor manually
        if preprocessor is not None:
            df = preprocessor.transform(df)
            proba = model.predict_proba(df)[:, 1]
        else:
            raise ValueError("Model cannot predict directly and no preprocessor was provided.")

    labels = (proba > 0.5).astype(int)

    results = [
        {"label": int(label), "probability": float(p)}
        for label, p in zip(labels, proba)
    ]
    return results
