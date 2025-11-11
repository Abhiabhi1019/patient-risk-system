# src/tests/test_pipeline.py
import os
import pandas as pd
from src.models.train import main as train_main
from src.models.predict import load_artifacts, predict_dataframe

# Get project root dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
processed_csv = os.path.join(PROJECT_ROOT, "data/processed/processed.csv")
output_dir = os.path.join(PROJECT_ROOT, "outputs")

def test_end_to_end():
    # 1️⃣ Ensure processed data exists
    assert os.path.exists(processed_csv), f"{processed_csv} not found! Run preprocessing first."

    # 2️⃣ Run training
    train_main(processed_csv, output_dir, "random_forest")
    model_path = os.path.join(output_dir, "model.joblib")
    assert os.path.exists(model_path), f"{model_path} not found! Training might have failed."

    # 3️⃣ Load artifacts
    model, preprocessor, expected_raw_columns = load_artifacts(output_dir)

    # 4️⃣ Load sample data (drop label)
    df = pd.read_csv(processed_csv).drop(columns=["readmitted"])

    # 5️⃣ Predict
    results = predict_dataframe(df, model=model, preprocessor=preprocessor)

    # 6️⃣ Check predictions
    assert len(results) == len(df), "Number of predictions does not match number of samples."
    for r in results:
        assert 0 <= r["probability"] <= 1, "Probability out of bounds."
        assert r["label"] in [0, 1], "Label should be 0 or 1."
