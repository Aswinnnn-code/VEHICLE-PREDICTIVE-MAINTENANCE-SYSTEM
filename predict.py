import streamlit as st
import joblib
import pandas as pd
import os


save_path = r"C:/placement/AI AUTO/model_files"
scaler = joblib.load(os.path.join(save_path, "scaler.pkl"))
model = joblib.load(os.path.join(save_path, "vehicle_maintenance_model.pkl"))


st.set_page_config(page_title="Vehicle Maintenance Predictor", page_icon="🚗", layout="centered")


st.markdown("<h1 style='text-align: center; color: #ff5733;'>🚗 Vehicle Maintenance Predictor</h1>", unsafe_allow_html=True)
st.write("Enter your vehicle parameters below to check if it needs maintenance.")


col1, col2 = st.columns(2)
with col1:
    mileage = st.number_input("Mileage (km)", min_value=0.0)
    engine_temp = st.number_input("Engine Temperature (°C)", min_value=0.0)
    rpm = st.number_input("RPM", min_value=0.0)
with col2:
    oil_pressure = st.number_input("Oil Pressure (psi)", min_value=0.0)
    fuel_efficiency = st.number_input("Fuel Efficiency (km/l)", min_value=0.0)

if st.button("🔍 Predict", use_container_width=True) :
    input_df = pd.DataFrame([{
        "mileage": mileage,
        "engine_temp": engine_temp,
        "rpm": rpm,
        "oil_pressure": oil_pressure,
        "fuel_efficiency": fuel_efficiency
    }])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ Needs Maintenance", icon="🚨")
    else:
        st.success("✅ No Maintenance Needed", icon="🛠️")

st.markdown("<hr><p style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
