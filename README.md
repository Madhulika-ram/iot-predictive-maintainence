# IoT Predictive Maintenance System

## Overview

This project simulates an IoT-based predictive maintenance system to monitor machine health using sensor data. It detects anomalies, predicts potential failures, and generates actionable alerts to prevent unexpected breakdowns.

---

## Problem Statement

In industrial environments, machines often fail unexpectedly due to:

* Overheating
* Excessive vibration
* Pressure fluctuations

This leads to:

* High downtime
* Increased maintenance cost
* Safety risks

---

## Solution

This system analyzes sensor data (temperature, vibration, pressure) to:

* Detect anomalies using machine learning
* Predict potential failures
* Generate intelligent alerts
* Visualize insights through an interactive dashboard

---

## Key Features

* Sensor data simulation (realistic IoT scenario)
* Anomaly detection using Isolation Forest
* Failure prediction using Random Forest
* Alert generation system for decision-making
* Interactive dashboard using Streamlit

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit

---

## How to Run

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate (PowerShell)

```bash
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Pipeline

```bash
python src/data_generator.py
python src/preprocessing.py
python src/anomaly_detection.py
python src/model.py
python src/alerts.py
```

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---
## Key Results: 
* 99.5% accuracy
* 0.95 F1-score
* 50 anomalies detected

---

## Results & Insights

* Temperature and vibration are the strongest indicators of machine failure
* Early anomaly detection can reduce downtime significantly
* Alert system provides actionable insights for maintenance decisions

Note: Current failure labels are derived from anomaly detection, leading to high model accuracy. In real-world scenarios, labels would come from actual failure logs.

---

## Future Improvements

* Integrate real IoT sensor data
* Add time-series forecasting (LSTM / ARIMA)
* Deploy using Streamlit Cloud
* Add GenAI-based automated explanations

---

## Use Cases

* Manufacturing plants
* Smart factories
* Power systems
* Industrial automation

---

## Dashboard Preview

<img width="1876" height="842" alt="SmartIoTPredictiveSystem" src="https://github.com/user-attachments/assets/edcda9c7-1052-40b7-9a12-db010e94934a" />


---

## Author

Ramaneti Madhulika
