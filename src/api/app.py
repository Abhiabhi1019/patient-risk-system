# src/api/app.py
from flask import Flask, request, jsonify
import pandas as pd
import yaml
from src.models.predict import predict_dataframe
from src.db.db_client import save_prediction, init_db

# ✅ Load config
with open("src/config/config.yaml") as f:
    config = yaml.safe_load(f)

app = Flask(__name__)

# ✅ Initialize database
init_db()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    preds = predict_dataframe(df)

    # ✅ Log predictions to DB
    for i, row in df.iterrows():
        patient_id = row.get("patient_id", f"PATIENT_{i+1}")
        prob = preds[i]["probability"]
        label = preds[i]["label"]
        save_prediction(patient_id=patient_id, probability=prob, label=label, metadata=str(row.to_dict()))

    return jsonify(preds)

if __name__ == "__main__":
    host = config["api"]["host"]
    port = config["api"]["port"]
    app.run(host=host, port=port)
