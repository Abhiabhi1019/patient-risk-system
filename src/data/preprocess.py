import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def preprocess_data(in_path, out_path, label_col):
    """Clean, impute, and scale/encode the dataset."""
    print(f"[INFO] Loading data from {in_path}")
    df = pd.read_csv(in_path)

    # Separate features and label
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"[INFO] Numeric columns: {num_cols}")
    print(f"[INFO] Categorical columns: {cat_cols}")

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
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )

    # Fit and transform the data
    print("[INFO] Fitting preprocessing pipeline...")
    X_processed = preprocessor.fit_transform(X)

    # Get feature names
    encoded_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(encoded_cols)

    # Convert to DataFrame
    processed_df = pd.DataFrame(X_processed, columns=feature_names)
    processed_df[label_col] = y.values

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save processed data
    processed_df.to_csv(out_path, index=False)
    print(f"[INFO] Processed data saved to {out_path}")

    # Save preprocessor
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(preprocessor, "outputs/preprocessor.joblib")
    print("[INFO] Preprocessor saved to outputs/preprocessor.joblib")

    return processed_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess patient data")
    parser.add_argument("--in_path", type=str, required=True, help="Input CSV path")
    parser.add_argument("--out_path", type=str, required=True, help="Output CSV path")
    parser.add_argument("--label_col", type=str, required=True, help="Target column name")
    args = parser.parse_args()

    preprocess_data(args.in_path, args.out_path, args.label_col)
