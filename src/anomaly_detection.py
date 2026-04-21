import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("data/processed/scaled_data.csv")

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df)

df.to_csv("data/processed/anomalies.csv", index=False)

print("Anomaly detection complete")