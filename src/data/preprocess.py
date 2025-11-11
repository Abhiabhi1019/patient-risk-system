import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
import json  # <- import json for saving feature schema

# Define the features we expect
NUM_FEATURES = ["age", "blood_pressure", "cholesterol"]
CAT_FEATURES = ["gender"]

def preprocess_data(in_path, out_path, label_col):
    """Clean, impute, and prepare the dataset while saving the preprocessor and schema."""
    print(f"[INFO] Loading data from {in_path}")
    df = pd.read_csv(in_path)

    # Keep only expected columns
    expected_cols = NUM_FEATURES + CAT_FEATURES + [label_col]
    df = df[expected_cols]

    # Separate features and label
    X = df.drop(columns=[label_col])
    y = df[label_col]

    print(f"[INFO] Numeric columns: {NUM_FEATURES}")
    print(f"[INFO] Categorical columns: {CAT_FEATURES}")

    # Numeric pipeline
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUM_FEATURES),
            ('cat', cat_pipeline, CAT_FEATURES)
        ]
    )

    # Fit preprocessor on training data
    print("[INFO] Fitting preprocessing pipeline...")
    preprocessor.fit(X)

    # Save original cleaned data (not encoded)
    processed_df = X.copy()
    processed_df[label_col] = y.values

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save cleaned data
    processed_df.to_csv(out_path, index=False)
    print(f"[INFO] Cleaned (raw) data saved to {out_path}")

    # Save preprocessor
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(preprocessor, "outputs/preprocessor.joblib")
    print("[INFO] Preprocessor saved to outputs/preprocessor.joblib")

    # Save feature schema for future predictions
    schema = {"raw_columns": NUM_FEATURES + CAT_FEATURES}
    with open("outputs/feature_schema.json", "w") as f:
        json.dump(schema, f)
    print("[INFO] Feature schema saved to outputs/feature_schema.json")

    return processed_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess patient data")
    parser.add_argument("--in_path", type=str, required=True, help="Input CSV path")
    parser.add_argument("--out_path", type=str, required=True, help="Output CSV path")
    parser.add_argument("--label_col", type=str, required=True, help="Target column name")
    args = parser.parse_args()

    preprocess_data(args.in_path, args.out_path, args.label_col)
