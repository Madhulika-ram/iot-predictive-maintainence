import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

rows = 1000
start_time = datetime.now()

timestamps = [start_time + timedelta(minutes=i) for i in range(rows)]

temperature = np.random.normal(30, 2, rows)
vibration = np.random.normal(5, 0.8, rows)
pressure = np.random.normal(100, 5, rows)

# Inject anomalies
for i in range(50):
    idx = np.random.randint(0, rows)
    temperature[idx] += np.random.uniform(10, 20)
    vibration[idx] += np.random.uniform(3, 5)

df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure
})

df.to_csv("data/raw/sensor_data.csv", index=False)
print("Data generated")