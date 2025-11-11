# src/db/db_client.py

import sqlite3
import json
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


# 1️⃣ Initialize the database
def init_db(db_uri="sqlite:///data/patient_risk.db"):
    """
    Initialize SQLite database and create table if not exists.
    db_uri: format 'sqlite:///path/to/db'
    """
    db_path = db_uri.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_data TEXT,
            prediction INTEGER,
            probability REAL
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"✅ Database initialized at {db_path}")


# 2️⃣ Save prediction results
def save_prediction(db_uri, input_data, prediction, probability):
    """
    Save a prediction entry to the database.
    """
    try:
        db_path = db_uri.replace("sqlite:///", "")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions (timestamp, input_data, prediction, probability)
            VALUES (?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), json.dumps(input_data), prediction, probability))

        conn.commit()
        conn.close()
        logger.info("✅ Prediction saved to database.")

    except Exception as e:
        logger.error(f"❌ Failed to save prediction: {e}")
