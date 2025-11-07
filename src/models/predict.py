"""
predict.py
Utilities to load preprocessor + model and run predictions on pandas DataFrames.
Handles missing columns gracefully.
"""

import joblib
import pandas as pd
import numpy as np
import os
from typing import List

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/model.pkl")
DEFAULT_PP_PATH = os.environ.get("PP_PATH", "outputs/preprocessor.joblib")

def load_artifacts(model_path: str = DEFAULT_MODEL_PATH, pp_path: str = DEFAULT_PP_PATH):
    """Load and return (model, preprocessor). Raises FileNotFoundError if missing."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(pp_path):
        raise FileNotFoundError(f"Preprocessor file not found: {pp_path}")
    model = joblib.load(model_path)
    preprocessor = joblib.load(pp_path)
    return model, preprocessor

def ensure_columns(df: pd.DataFrame, preprocessor):
    """Ensure df has all required columns used during training."""
    if hasattr(preprocessor, "feature_names_in_"):
        required = list(preprocessor.feature_names_in_)
    elif hasattr(preprocessor, "transformers_"):
        # For ColumnTransformer, infer from transformers
        required = []
        for _, trans, cols in preprocessor.transformers_:
            if isinstance(cols, list):
                required.extend(cols)
    else:
        return df

    missing = [c for c in required if c not in df.columns]
    for c in missing:
        df[c] = np.nan  # fill missing with NaN
    if missing:
        print(f"⚠️ Warning: Added missing columns with NaN: {missing}")
    return df[required]

def predict_dataframe(df: pd.DataFrame, model=None, preprocessor=None,
                      model_path: str = DEFAULT_MODEL_PATH, pp_path: str = DEFAULT_PP_PATH):
    """
    Predict readmission probabilities for a dataframe of raw features.
    Returns a list of dicts: [{'probability': float, 'label': int}, ...]
    """
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts(model_path, pp_path)

    df = ensure_columns(df, preprocessor)
    X_trans = preprocessor.transform(df)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_trans)[:, 1]
    else:
        # Try xgboost booster predict path
        try:
            import xgboost as xgb
            if isinstance(model, xgb.Booster):
                dmat = xgb.DMatrix(X_trans)
                probs = model.predict(dmat)
            else:
                probs = model.predict(X_trans)
        except Exception:
            probs = model.predict(X_trans)

    results = [{"probability": float(p), "label": int(p > 0.5)} for p in probs]
    return results

def predict_from_records(records: List[dict], **kwargs):
    """Accepts list of JSON-like records, builds DataFrame, returns predictions."""
    df = pd.DataFrame.from_records(records)
    return predict_dataframe(df, **kwargs)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sample_path = sys.argv[1]
        df = pd.read_csv(sample_path)
        res = predict_dataframe(df)
        print(res)
    else:
        print("Usage: python src/models/predict.py data/sample.csv")
