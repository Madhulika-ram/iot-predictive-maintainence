import pandas as pd

df = pd.read_csv("data/processed/anomalies.csv")

def generate_alert(row):
    if row['anomaly'] == -1:
        if row['temperature'] > 40:
            return "High temperature → Overheating risk"
        elif row['vibration'] > 8:
            return "High vibration → Mechanical fault"
        elif row['pressure'] > 110:
            return "Pressure spike → System instability"
        else:
            return "General anomaly detected"
    return "Normal"

df['alert'] = df.apply(generate_alert, axis=1)

df.to_csv("data/processed/alerts.csv", index=False)

print("Alerts generated successfully")