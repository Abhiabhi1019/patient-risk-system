# src/tests/test_model.py
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_load():
    model = joblib.load("outputs/model.pkl")
    assert model is not None, "Model not loaded"

def test_model_prediction():
    df = pd.read_csv("data/processed/processed.csv").head(5)
    model = joblib.load("outputs/model.pkl")
    preds = model.predict(df.drop(columns=["readmitted"]))
    assert len(preds) == len(df), "Prediction size mismatch"
