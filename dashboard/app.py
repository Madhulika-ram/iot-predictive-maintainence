import streamlit as st
import pandas as pd

df = pd.read_csv('F:/projectssss/iot-predictive-maintenance/data/processed/alerts.csv')

st.set_page_config(page_title="Predictive Maintenance", layout="wide") # set page title and layout

st.title("Smart IoT Predictive Maintenance System") # main title

# Metrics
total = len(df)
anomalies = df[df['anomaly'] == -1] # filter anomalies based on the 'anomaly' column where -1 indicates an anomaly

col1, col2 = st.columns(2) # display metrics in two columns
col1.metric("Total Records", total) # display total records
col2.metric("Anomalies Detected", len(anomalies)) # display number of anomalies detected

# Charts
st.subheader("Sensor Trends") # line chart to show trends in sensor data
st.line_chart(df[['temperature','vibration','pressure']]) 

# Alerts table
st.subheader("Anomaly Alerts") # display anomalies in a table format
st.dataframe(anomalies[['temperature','vibration','pressure','alert']]) # display only relevant columns in the anomalies table

# Filter option
st.subheader("Filter Data")
option = st.selectbox("Show Data", ["All", "Only Anomalies"]) # dropdown to filter between all data and only anomalies

if option == "Only Anomalies": 
    st.write(anomalies) # display anomalies dataframe
else:
    st.write(df) # display full dataframe