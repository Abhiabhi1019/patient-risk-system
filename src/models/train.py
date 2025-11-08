# src/models/train.py
from src.utils.logger import get_logger
import pandas as pd
import joblib
import json
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def load_config():
    """Load YAML configuration."""
    with open("src/config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data(path):
    """Load processed dataset."""
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)

def build_pipeline(categorical_features, numeric_features):
    """Build ML preprocessing + model pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline

def train_and_evaluate(config):
    """Train model and evaluate metrics."""
    df = load_data(config["data"]["processed_path"])
    target = config["model"]["target"]

    X = df.drop(columns=[target])
    y = df[target]

    # Separate categorical and numerical columns
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["train"]["test_size"],
        random_state=config["train"]["random_state"]
    )

    logger.info("Building pipeline...")
    pipeline = build_pipeline(categorical, numerical)

    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_proba), 4),
        "timestamp": datetime.utcnow().isoformat()
    }

    logger.info(f"Metrics: {metrics}")

    # Save model & metrics
    os.makedirs("outputs", exist_ok=True)
    model_path = config["model"]["save_path"]
    joblib.dump(pipeline, model_path)

    with open(config["model"]["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metrics saved to {config['model']['metrics_path']}")
    return metrics

if __name__ == "__main__":
    config = load_config()
    metrics = train_and_evaluate(config)
    print("✅ Training complete. Metrics:")
    print(json.dumps(metrics, indent=4))
