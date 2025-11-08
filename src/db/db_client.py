from src.db.db_client import save_prediction, init_db

# Initialize DB table at startup
init_db()


"""
Simple DB client to store prediction logs. Default: SQLite local file `predictions.db`.
Change DB_URL env var to point to a different DB (e.g. postgres).
"""

import os
from sqlalchemy import create_engine, text
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use .env or default SQLite
DB_URL = os.environ.get("DB_URL", "sqlite:///data/patient_data.db")

# Initialize DB engine
engine = create_engine(DB_URL, echo=False)

def init_db():
    """Create the predictions table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                probability REAL,
                label INTEGER,
                metadata TEXT,
                created_at TEXT
            )
        """))
    print("✅ Database initialized:", DB_URL)

def save_prediction(patient_id: str, probability: float, label: int, metadata: str = None):
    """Insert a prediction log into the database."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO predictions (patient_id, probability, label, metadata, created_at)
                VALUES (:pid, :prob, :lab, :meta, :ts)
            """),
            {
                "pid": patient_id,
                "prob": float(probability),
                "lab": int(label),
                "meta": metadata or "",
                "ts": datetime.utcnow().isoformat()
            }
        )
    print(f"📝 Prediction saved for patient {patient_id}")

if __name__ == "__main__":
    init_db()
