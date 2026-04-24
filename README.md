# Smart IoT Predictive Maintenance System

> Simulates an industrial IoT environment to detect sensor anomalies, predict machine failures, and generate actionable maintenance alerts — all visualised through an interactive dashboard.
<img width="1792" height="860" alt="Sensor_Trends" src="https://github.com/user-attachments/assets/9250845a-ec7b-47f2-821d-23fa43e33094" />
<img width="1786" height="547" alt="Anamoly_alert_breakdowns" src="https://github.com/user-attachments/assets/7ecd0287-9a32-4d6b-8764-3f40c5ea205b" />
<img width="1792" height="612" alt="Anamoly_alerts_detail" src="https://github.com/user-attachments/assets/18072c4b-e4f1-4e46-9ead-0d77ea58e415" />
<img width="1813" height="765" alt="Explore_data" src="https://github.com/user-attachments/assets/00657fcd-415a-4e0a-8fa1-312287545011" />

## Problem Statement

In industrial environments, machines fail unexpectedly due to overheating, excessive vibration, and pressure fluctuations. This leads to costly downtime, increased maintenance spend, and safety risks. Traditional rule-based monitoring reacts after damage occurs.

This project takes a predictive approach — analysing live sensor streams to catch anomalies before they become failures.

---

## Solution Overview

The system analyses three sensor channels — **temperature**, **vibration**, and **pressure** — through a machine learning pipeline:

1. Detects anomalies using **Isolation Forest** (unsupervised)
2. Predicts failures using **Random Forest** trained on ground-truth labels
3. Classifies each anomaly into a specific alert type (overheating, mechanical fault, pressure spike)
4. Visualises everything through an interactive **Streamlit dashboard**

---

## Project Structure

```
iot-predictive-maintenance/
├── src/
│   ├── data_generator.py       # Simulates sensor data with injected failures
│   ├── preprocessing.py        # StandardScaler + saves scaler.pkl
│   ├── anomaly_detection.py    # Isolation Forest anomaly flagging
│   ├── alerts.py               # Rule-based alert classification
│   ├── model.py                # Random Forest failure prediction
│   └── run_pipeline.py         # Single-command pipeline runner
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── data/
│   ├── raw/                    # Generated sensor data
│   └── processed/              # Scaled data, anomalies, alerts, models
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python, NumPy, Pandas, scikit-learn |
| Visualisation | Streamlit, Matplotlib, Seaborn |
| Model persistence | joblib |
| Environment | venv |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/iot-predictive-maintenance.git
cd iot-predictive-maintenance
```

### 2. Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1     # PowerShell
# or
venv\Scripts\activate.bat       # Command Prompt
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python src/run_pipeline.py
```

This runs all four stages in order — data generation → preprocessing → anomaly detection → alert generation — and stops immediately with a clear error message if any step fails.

### 5. Train the prediction model

```bash
python src/model.py
```

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Opens automatically at `http://localhost:8501`.

---

## Pipeline Stages

```
data_generator.py
      │  1000 rows of sensor data + ground-truth failure labels
      ▼
preprocessing.py
      │  StandardScaler → scaled_data.csv + scaler.pkl
      ▼
anomaly_detection.py
      │  Isolation Forest → anomalies.csv (anomaly flag + raw sensor values)
      ▼
alerts.py
      │  Rule-based classification → alerts.csv
      ▼
model.py
      │  Random Forest on ground-truth labels → rf_model.pkl
      ▼
dashboard/app.py
         Streamlit visualisation
```

---

## How It Works

### Anomaly Detection — Isolation Forest

Isolation Forest is an unsupervised algorithm that isolates anomalies by randomly partitioning the feature space. Anomalous points require fewer splits to isolate, giving them a lower anomaly score. No labels are needed — it learns purely from the data distribution.

`contamination=0.05` tells the model to expect approximately 5% anomalies across 1000 readings.

### Failure Prediction — Random Forest

Random Forest is a supervised ensemble model trained on the `actual_failure` column — ground-truth labels from the data generator, not derived from Isolation Forest output. This avoids label leakage, where a model trained on its own predictions will always score artificially high.

`class_weight='balanced'` handles the 950:50 class imbalance between normal and failure records.

### Alert Classification

Each anomaly is classified into one of four types by comparing real sensor values (in original units) against physics-based thresholds:

| Alert | Condition | Meaning |
|---|---|---|
| High temperature | temp_raw > 40°C | Overheating risk |
| High vibration | vibration_raw > 8 mm/s | Mechanical fault |
| Pressure spike | pressure_raw > 115 bar | System instability |
| General anomaly | None of the above | Pattern-based outlier |

---

## Dashboard Features

- **4 KPI cards** — total records, anomalies detected, anomaly rate, top alert type
- **Sensor trend chart** — temperature, vibration, pressure over time in real units
- **Alert breakdown chart** — distribution of alert types
- **Anomaly alerts table** — filterable list of flagged records with sensor readings
- **Data explorer** — filter between all records, anomalies only, or normal only

---

## Key Design Decisions

**Why Isolation Forest for anomaly detection?**
It is unsupervised — no labels required — and well-suited for multivariate tabular data with a small anomaly fraction. It is also computationally efficient and interpretable.

**Why Random Forest for failure prediction?**
Tree-based models handle non-linear feature interactions naturally, are robust to outliers, and provide feature importance scores out of the box. The ensemble approach reduces overfitting on a small dataset.

**Why save the scaler?**
The `scaler.pkl` file ensures that any new sensor data is scaled consistently with the training distribution. Without it, live scoring would use different statistics and produce unreliable predictions.

---

## Results & Insights

- Temperature and vibration are the strongest predictors of failure (see `Feature_Importance.png`)
  <img width="600" height="400" alt="Featureimportance" src="https://github.com/user-attachments/assets/a2778f0a-3873-45b5-9b63-be7cd1229b20" />
- Pressure contributes less than 10% to model decisions in this simulated dataset
- Early anomaly detection flags failures before they reach critical thresholds

> **Note:** Failure labels in this project are synthetically injected during data generation. In a real deployment, labels would come from maintenance logs or engineer-tagged failure events.

---

## Future Improvements

- [ ] Integrate real IoT sensor streams (MQTT / Kafka)
- [ ] Add time-series forecasting (LSTM / ARIMA) for failure horizon prediction
- [ ] Deploy to Streamlit Cloud
- [ ] Add GenAI-powered alert explanations
- [ ] Extend to multi-machine monitoring with per-device anomaly profiles

---

## Use Cases

- Manufacturing plant monitoring
- Smart factory predictive maintenance
- Power generation systems
- Industrial automation pipelines

---

## Author

**Ramaneti Madhulika**  
[GitHub](https://github.com/Madhulika-ram) · [LinkedIn](https://linkedin.com/in/ramaneti-madhulika)
