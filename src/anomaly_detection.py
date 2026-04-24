import pandas as pd
from sklearn.ensemble import IsolationForest
from pathlib import Path
# Detect anomalies in the preprocessed sensor data using Isolation Forest and save the results to a new CSV file
def detect_anomalies(input_path: str = "data/processed/scaled_data.csv",
                     output_dir: str = "data/processed",
                     contamination: float = 0.05) -> pd.DataFrame:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    feature_cols = ['temperature', 'vibration', 'pressure']

    model = IsolationForest(contamination=contamination, random_state=42) # initialize the Isolation Forest model with the specified contamination level and random state for reproducibility
    df['anomaly'] = model.fit_predict(df[feature_cols])

    raw_path = Path("data/raw/sensor_data.csv") # define the path to the original raw data file to check if it exists for adding raw sensor readings to the anomalies DataFrame for reference
    # if the original raw data file exists, read it and add the original sensor readings to the anomalies DataFrame for referenc
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        df['temperature_raw'] = raw_df['temperature'].values
        df['vibration_raw'] = raw_df['vibration'].values
        df['pressure_raw'] = raw_df['pressure'].values

    df.to_csv(output_path / "anomalies.csv", index=False)
# Count the number of anomalies detected and print a summary message indicating the number and percentage of anomalies flagged in the dataset
    n_anomalies = (df['anomaly'] == -1).sum()
    print(f"Anomaly detection complete — {n_anomalies} anomalies flagged ({n_anomalies/len(df)*100:.1f}%)")
    return df
# Execute the anomaly detection function when the script is run directly
if __name__ == "__main__":
    detect_anomalies()