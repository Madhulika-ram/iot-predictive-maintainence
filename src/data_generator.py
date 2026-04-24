#import libraries
import pandas as pd # for data manipulation
import numpy as np # for numerical operations
from datetime import datetime, timedelta # for handling timestamps
from pathlib import Path # for handling file paths

np.random.seed(42) # for reproducibility of results

ROWS = 1000
START_TIME = datetime.now()
# function to generate synthetic sensor data for predictive maintenance
def generate_sensor_data() -> pd.DataFrame: 
    timestamps = [START_TIME + timedelta(minutes=pos) for pos in range(ROWS)] # generate timestamps at 1-minute intervals

    temperature = np.random.normal(30, 2, ROWS) # generate temperature data with mean 30 and std deviation 2
    vibration = np.random.normal(5, 0.8, ROWS) # generate vibration data with mean 5 and std deviation 0.8
    pressure = np.random.normal(100, 5, ROWS) # generate pressure data with mean 100 and std deviation 5

    failure_indices = set() # to keep track of indices where failures occur
    # randomly select 50 indices to simulate failures by adding significant deviations to the sensor readings
    for _ in range(50):
        idx = np.random.randint(0, ROWS) 
        failure_indices.add(idx) # add the index to the set of failure indices
        temperature[idx] += np.random.uniform(10, 20) # add a significant increase to temperature to simulate failure
        vibration[idx] += np.random.uniform(3, 5) # add a significant increase to vibration to simulate failure'
        pressure[idx] += np.random.uniform(15, 30) # add a significant increase to pressure to simulate failure

    failure_labels = [1 if pos in failure_indices else 0 for pos in range(ROWS)] # create a list of failure labels where 1 indicates a failure and 0 indicates normal operation
# create a DataFrame to hold the generated sensor data and failure labels
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "actual_failure": failure_labels
    })
    return df
# The main block to execute the data generation and save it to a CSV file
if __name__ == "__main__":
    output_path = Path("data/raw")
    output_path.mkdir(parents=True, exist_ok=True)

    df = generate_sensor_data()
    df.to_csv(output_path / "sensor_data.csv", index=False)
    print(f"Data generated: {len(df)} rows, {df['actual_failure'].sum()} labelled failures")