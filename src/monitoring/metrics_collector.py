import os
import json
import time
import psutil
from datetime import datetime

METRICS_FILE = "data/metrics.json"

def collect_system_metrics():
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
    }

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Load existing metrics
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            try:
                existing = json.load(f)
                if not isinstance(existing, list):  # ✅ fix for dict case
                    existing = []
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []

    # Add new entry
    existing.append(metrics)

    # Save
    with open(METRICS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"✅ Metrics collected at {metrics['timestamp']}")

if __name__ == "__main__":
    print("🔍 Monitoring system performance every 60 seconds...")
    while True:
        collect_system_metrics()
        time.sleep(60)
