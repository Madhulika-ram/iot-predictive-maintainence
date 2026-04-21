#import libraries
import pandas as pd # data manipulation
import numpy as np # numerical operations
from datetime import datetime, timedelta # for timestamps

np.random.seed(42) # set seed for reproducibility

rows = 1000
start_time = datetime.now() # current time as start point

timestamps = [start_time + timedelta(minutes=i) for i in range(rows)] # generate timestamps at 1-minute intervals

temperature = np.random.normal(30, 2, rows) # generate temperature data with mean 30 and std 2
vibration = np.random.normal(5, 0.8, rows) # generate vibration data with mean 5 and std 0.8
pressure = np.random.normal(100, 5, rows) # generate pressure data with mean 100 and std 5

# Inject anomalies
for i in range(50):
    idx = np.random.randint(0, rows) # randomly select an index to inject anomaly
    temperature[idx] += np.random.uniform(10, 20) # add random anomaly to temperature
    pressure[idx] += np.random.uniform(0.1, 0.5) # add random anomaly to pressure
    vibration[idx] += np.random.uniform(3, 5) # add random anomaly to vibration
# Create DataFrame and save to CSV
df = pd.DataFrame({
    "timestamp": timestamps, 
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure
})

df.to_csv("data/raw/sensor_data.csv", index=False)
print("Data generated")