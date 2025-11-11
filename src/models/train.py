import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from src.utils.logger import get_logger

logger = get_logger("train")


def train_model(input_csv="data/raw/patient_records.csv", model_type="logistic_regression"):
    logger.info(f"🚀 Starting model training using {model_type}...")

    # ✅ Load dataset
    df = pd.read_csv(input_csv)

    # ✅ Ensure 'patient_id' column exists
    if "patient_id" not in df.columns:
        df["patient_id"] = range(1, len(df) + 1)
        logger.warning("⚠️ 'patient_id' not found — created sequential IDs automatically.")

    # ✅ Define features and target
    feature_cols = ["patient_id", "age", "blood_pressure", "cholesterol", "gender"]
    target_col = "readmitted"

    # Validate columns
    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"columns are missing: {set(missing_cols)}")

    X = df[feature_cols]
    y = df[target_col]

    # ✅ Define preprocessing
    numeric_features = ["age", "blood_pressure", "cholesterol"]
    categorical_features = ["gender"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    # ✅ Select classifier
    if model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # ✅ Build pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # ✅ Train and evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info(f"✅ Model trained successfully. Accuracy: {acc:.2f}")

    return model, preprocessor, {"accuracy": acc}


def save_artifacts(model, preprocessor, output_dir="artifacts"):
    """Save trained model and preprocessor."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model.joblib"))
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    logger.info(f"✅ Artifacts saved in {output_dir}/")


def load_artifacts(output_dir="artifacts", *args, **kwargs):
    """Load model and preprocessor — return 3 items for compatibility."""
    model_path = os.path.join(output_dir, "model.joblib")
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Artifacts not found in {output_dir}. Please train the model first.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    logger.info(f"✅ Loaded model and preprocessor from {output_dir}/")
    return model, preprocessor, {}  # <-- always return 3 values


def main(input_csv="data/raw/patient_records.csv", output_dir="artifacts", model_type="logistic_regression"):
    model, preprocessor, metrics = train_model(input_csv, model_type)
    save_artifacts(model, preprocessor, output_dir)
    logger.info("🏁 Training pipeline finished successfully.")


if __name__ == "__main__":
    main()
