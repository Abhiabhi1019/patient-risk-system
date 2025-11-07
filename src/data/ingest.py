import os
import pandas as pd
import boto3

def download_from_s3(bucket_name, file_key, local_path):
    """
    Download raw patient data from S3 to local data/raw folder.
    """
    s3 = boto3.client('s3')
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket_name, file_key, local_path)
    print(f"✅ Downloaded {file_key} from S3 bucket {bucket_name} to {local_path}")

def load_local_data(local_path):
    """
    Load CSV data into a pandas DataFrame.
    """
    df = pd.read_csv(local_path)
    print(f"📄 Loaded {df.shape[0]} rows and {df.shape[1]} columns from {local_path}")
    return df

if __name__ == "__main__":
    # Example usage (local testing)
    # You can skip S3 and just use local CSV for now
    data_path = "data/raw/patient_records.csv"

    # if your file already exists locally:
    if os.path.exists(data_path):
        df = load_local_data(data_path)
    else:
        print("⚠️ No local data found. Please place a CSV in data/raw/ or configure S3.")
