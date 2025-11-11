import pandas as pd
import joblib

def test_model_prediction():
    # Load trained pipeline (which includes preprocessor + model)
    model = joblib.load("outputs/model.joblib")

    # ✅ Raw sample (each column has one value)
    sample_data = pd.DataFrame({
        "age": [45],
        "bmi": [28.3],
        "gender": ["F"],
        "smoking_status": ["never"],
        "blood_pressure": [120],
        "cholesterol": [190]
    })

    # ✅ Directly predict using the pipeline
    preds = model.predict(sample_data)

    # ✅ Verify
    assert len(preds) == 1
