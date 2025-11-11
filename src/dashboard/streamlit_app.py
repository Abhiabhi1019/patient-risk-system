import streamlit as st
import pandas as pd
import altair as alt
from src.models.predict import load_artifacts, predict_dataframe

# ===========================
# 🩺 Streamlit Setup
# ===========================
st.set_page_config(page_title="Patient Risk Prediction Dashboard", layout="wide")
st.title("🏥 Patient Readmission Risk Prediction Dashboard")

st.markdown("""
This dashboard predicts **patient readmission risk** and provides both  
single-patient predictions and **batch analysis with visualization**.
""")

# ===========================
# 🔧 Load Model & Preprocessor
# ===========================
model, preprocessor, expected_raw_columns = load_artifacts()

# ===========================
# 🧍 Single Prediction
# ===========================
st.sidebar.header("🧍 Single Patient Input")

age = st.sidebar.slider("Age", 18, 100, 55)
blood_pressure = st.sidebar.number_input("Blood Pressure", 80, 200, 130)
cholesterol = st.sidebar.number_input("Cholesterol", 100, 300, 210)
gender = st.sidebar.selectbox("Gender", ["M", "F"])

input_df = pd.DataFrame([{
    "age": age,
    "blood_pressure": blood_pressure,
    "cholesterol": cholesterol,
    "gender": gender
}])

st.subheader("📋 Input Data Preview")
st.dataframe(input_df)

if st.button("🔮 Predict for Single Patient"):
    try:
        result = predict_dataframe(input_df, model, preprocessor, expected_raw_columns)
        label = result[0]["label"]
        probability = result[0]["probability"]

        st.metric(
            label="Predicted Readmission Risk",
            value=f"{probability*100:.2f}%",
            delta="High Risk" if label == 1 else "Low Risk"
        )

        if label == 1:
            st.warning("⚠️ The patient is at HIGH risk of readmission.")
        else:
            st.success("✅ The patient is at LOW risk of readmission.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===========================
# 📊 Batch Prediction Section
# ===========================
st.markdown("---")
st.header("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload a CSV file with patient data", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ File successfully uploaded.")
        st.dataframe(df.head())

        # Predict for all patients in uploaded CSV
        results = predict_dataframe(df, model, preprocessor, expected_raw_columns)
        results_df = pd.DataFrame(results)
        output_df = pd.concat([df, results_df], axis=1)

        st.subheader("🧠 Prediction Results")
        st.dataframe(output_df)

        # ===========================
        # 📈 Visualization
        # ===========================
        st.subheader("📊 Risk Probability Distribution")

        chart = (
            alt.Chart(output_df)
            .mark_bar()
            .encode(
                x=alt.X("probability:Q", bin=alt.Bin(maxbins=20), title="Predicted Probability"),
                y=alt.Y("count()", title="Number of Patients"),
                color=alt.Color("label:N", scale=alt.Scale(scheme="redblue"), title="Predicted Class")
            )
            .properties(width=700, height=400)
        )

        st.altair_chart(chart, use_container_width=True)

        st.download_button(
            label="💾 Download Predictions as CSV",
            data=output_df.to_csv(index=False),
            file_name="batch_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# ===========================
# 📘 Footer
# ===========================
st.markdown("""
---
📅 **Developed by:** Healthcare AI Team  
📈 **Features:** Real-time predictions, batch processing, and visualization  
© 2025 Patient Risk Prediction System
""")
