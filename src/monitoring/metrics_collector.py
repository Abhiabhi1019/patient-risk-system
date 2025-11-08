# src/monitoring/metrics_collector.py
import json
import time
from datetime import datetime
import psutil
import os

def collect_system_metrics(output_path="src/models/metrics.json"):
    """Collects system-level metrics like CPU, memory, and disk usage."""
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage("/").percent
    }

    # Load previous metrics if available
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing = []
    else:
        existing = []

    # Append and write back
    existing.append(metrics)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=4)

if __name__ == "__main__":
    print("🔍 Monitoring system performance every 60 seconds...")
    while True:
        collect_system_metrics()
        time.sleep(60)
