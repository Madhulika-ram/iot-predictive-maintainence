import streamlit as st
import pandas as pd

df = pd.read_csv("F:/projectssss/iot-predictive-maintenance/data/processed/anomalies.csv")

st.title("IoT Predictive Maintenance Dashboard")

st.subheader("Sensor Trends")
st.line_chart(df[['temperature','vibration','pressure']])

st.subheader("Detected Anomalies")
st.write(df[df['anomaly'] == -1])