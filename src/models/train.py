import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

DATA_PATH = "data/processed/processed.csv"
MODEL_PATH = "outputs/model.pkl"

def train_model():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    # Save model
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
