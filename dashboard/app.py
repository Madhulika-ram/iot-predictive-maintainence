import streamlit as st
import pandas as pd

df = pd.read_csv('F:/projectssss/iot-predictive-maintenance/data/processed/alerts.csv')

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

st.title("Smart IoT Predictive Maintenance System")

# Metrics
total = len(df)
anomalies = df[df['anomaly'] == -1]

col1, col2 = st.columns(2)
col1.metric("Total Records", total)
col2.metric("Anomalies Detected", len(anomalies))

# Charts
st.subheader("Sensor Trends")
st.line_chart(df[['temperature','vibration','pressure']])

# Alerts table
st.subheader("Anomaly Alerts")
st.dataframe(anomalies[['temperature','vibration','pressure','alert']])

# Filter option
st.subheader("Filter Data")
option = st.selectbox("Show Data", ["All", "Only Anomalies"])

if option == "Only Anomalies":
    st.write(anomalies)
else:
    st.write(df)