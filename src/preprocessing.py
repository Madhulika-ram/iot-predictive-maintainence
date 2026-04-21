import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/raw/sensor_data.csv")

features = df[['temperature','vibration','pressure']] # select only sensor columns for scaling

scaler = StandardScaler() # initialize scaler
scaled = scaler.fit_transform(features) # fit and transform the data

scaled_df = pd.DataFrame(scaled, columns=features.columns) # add timestamp back to the scaled dataframe
scaled_df.to_csv("data/processed/scaled_data.csv", index=False) 

print("Preprocessing done")