import os
from dotenv import load_dotenv
import json
import logging

# Load environment variables
load_dotenv()

METRICS_PATH = os.getenv("METRICS_PATH", "src/models/metrics.json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

def check_model_performance():
    """Check model performance metrics and trigger warnings if below threshold."""
    if not os.path.exists(METRICS_PATH):
        logger.warning("⚠️ Metrics file not found.")
        return

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    acc = metrics.get("accuracy", None)
    if acc is None:
        logger.warning("⚠️ Accuracy metric missing from metrics.json.")
        return

    logger.info(f"📈 Model Accuracy: {acc:.2f}")

    if acc < 0.75:
        logger.warning("⚠️ Accuracy below threshold! Retrain recommended.")
	
