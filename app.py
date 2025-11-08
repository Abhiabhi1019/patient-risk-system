from flask import Flask, request, jsonify
import pandas as pd
from src.models.predict import predict_dataframe
from src.db.db_client import save_prediction, init_db

app = Flask(__name__)

# Initialize DB once at startup
init_db()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    preds = predict_dataframe(df)

    # Log each prediction to the database
    for i, row in df.iterrows():
        patient_id = row.get("patient_id", f"PATIENT_{i+1}")
        prob = preds[i]["probability"]
        label = preds[i]["label"]
        save_prediction(patient_id=patient_id, probability=prob, label=label, metadata=str(row.to_dict()))

    return jsonify(preds)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
