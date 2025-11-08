import os
import pandas as pd
from src.models.train import main as train_main
from src.models.predict import predict_dataframe

def test_end_to_end():
    assert os.path.exists("data/processed/processed.csv")
    df = pd.read_csv("data/processed/processed.csv")
    train_main("data/processed/processed.csv", "outputs", "random_forest")
    assert os.path.exists("outputs/model.pkl")
    df = df.drop(columns=["readmitted"])
    preds = predict_dataframe(df)
    assert len(preds) == len(df)
    print("✅ End-to-end test passed.")
