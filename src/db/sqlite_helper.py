import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("DB_URI", "sqlite:///data/patient_data.db")

def get_connection():
    db_path = DB_URI.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    return conn
