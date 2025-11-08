# src/dashboard/streamlit_app.py
import streamlit as st
import pandas as pd
import json
import os
import joblib
from src.models.predict import predict_dataframe

# -----------------------------
# 🎨 Page configuration
# -----------------------------
st.set_page_config(
    page_title="🧠 Patient Risk Prediction Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Patient Readmission Risk Prediction System")
st.markdown("This dashboard provides real-time analytics and predictions for patient readmission risk.")

# -----------------------------
# 🧩 Load model & metrics
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "outputs/model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please train the model first.")
        return None
    return joblib.load(model_path)

@st.cache_data
def load_metrics():
    metrics_path = "outputs/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}

model = load_model()
metrics = load_metrics()

# -----------------------------
# 📊 Metrics Section
# -----------------------------
st.header("📈 Model Performance")

if metrics:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", metrics.get("accuracy", 0))
    col2.metric("Precision", metrics.get("precision", 0))
    col3.metric("Recall", metrics.get("recall", 0))
    col4.metric("F1 Score", metrics.get("f1_score", 0))
    col5.metric("AUC", metrics.get("auc", 0))
else:
    st.warning("⚠️ No metrics found. Train the model to generate metrics.json")

# -----------------------------
# 🧍 Prediction Section
# -----------------------------
st.header("🔮 Predict Patient Readmission Risk")

option = st.radio("Choose input method:", ["Upload CSV", "Manual Entry"])

if option == "Upload CSV":
    uploaded_file = st.file_uploader("📁 Upload patient data (CSV format)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📋 Uploaded Data Preview:", df.head())

        if st.button("Predict Readmission Risk"):
            with st.spinner("Predicting..."):
                results = predict_dataframe(df)
                results_df = pd.DataFrame(results)
                df["probability"] = results_df["probability"]
                df["predicted_label"] = results_df["label"]

                st.success("✅ Predictions complete!")
                st.write(df.head())
                st.download_button("⬇️ Download Predictions", df.to_csv(index=False).encode('utf-8'), "predictions.csv")

elif option == "Manual Entry":
    st.subheader("Enter Patient Details")
    
    age = st.slider("Age", 0, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
    glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=180)
    heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=80)

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "blood_pressure": blood_pressure,
            "glucose": glucose,
            "cholesterol": cholesterol,
            "heart_rate": heart_rate
        }])

        with st.spinner("Running prediction..."):
            result = predict_dataframe(input_df)
            prob = result[0]["probability"]
            label = result[0]["label"]

            st.success(f"✅ Predicted Probability of Readmission: **{prob*100:.2f}%**")
            if label == 1:
                st.error("⚠️ High Risk of Readmission")
            else:
                st.info("✅ Low Risk of Readmission")

# -----------------------------
# 🧾 Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, scikit-learn, and Python.")
