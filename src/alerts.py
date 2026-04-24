import pandas as pd
from pathlib import Path
# Define thresholds for generating specific alert messages based on raw sensor readings when an anomaly is detected
TEMP_HIGH = 40
VIBRATION_HIGH = 8
PRESSURE_HIGH = 115
# Generate a specific alert message based on the raw sensor readings for a given row of data when an anomaly is detected
def generate_alert(row: pd.Series) -> str:
    if row['anomaly'] != -1:
        return "Normal"

    temp = row.get('temperature_raw', None)
    vib = row.get('vibration_raw', None)
    pres = row.get('pressure_raw', None)
# Check the raw sensor readings against defined thresholds to determine the specific type of alert to generate for the detected anomaly, prioritizing temperature, then vibration, and finally pressure
    if temp is not None and temp > TEMP_HIGH:
        return f"High temperature ({temp:.1f}°C) → Overheating risk"
    elif vib is not None and vib > VIBRATION_HIGH:
        return f"High vibration ({vib:.1f} mm/s) → Mechanical fault"
    elif pres is not None and pres > PRESSURE_HIGH:
        return f"Pressure spike ({pres:.1f} bar) → System instability"
    else:
        return "General anomaly detected"
# Generate specific alert messages for each row in the anomalies DataFrame based on the raw sensor readings and save the results to a new CSV file, while also printing a summary of the generated alerts
def generate_alerts(input_path: str = "data/processed/anomalies.csv",
                    output_path: str = "data/processed/alerts.csv") -> pd.DataFrame:

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df['alert'] = df.apply(generate_alert, axis=1)

    df.to_csv(output_path, index=False)
# Count the number of each type of alert generated and print a summary message indicating the distribution of alerts in the dataset
    alert_counts = df[df['anomaly'] == -1]['alert'].value_counts()
    print("Alerts generated:\n", alert_counts.to_string())
    return df
# Execute the alert generation function when the script is run directly
if __name__ == "__main__":
    generate_alerts()