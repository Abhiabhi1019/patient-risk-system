import json
import time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="System Metrics Dashboard", layout="wide")
st.title("📊 System Performance Monitoring")

METRICS_FILE = "data/metrics.json"

def load_metrics():
    try:
        with open(METRICS_FILE, "r") as f:
            data = json.load(f)
            return pd.DataFrame(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return pd.DataFrame()

placeholder = st.empty()

while True:
    df = load_metrics()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp', ascending=True)
        
        with placeholder.container():
            st.line_chart(df.set_index('timestamp')[['cpu_percent', 'memory_percent', 'disk_percent']])
            st.dataframe(df.tail(5))
    else:
        st.warning("No metrics data found yet.")
    
    time.sleep(5)
