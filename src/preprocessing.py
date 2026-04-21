import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/raw/sensor_data.csv")

features = df[['temperature','vibration','pressure']]

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

scaled_df = pd.DataFrame(scaled, columns=features.columns)
scaled_df.to_csv("data/processed/scaled_data.csv", index=False)

print("Preprocessing done")