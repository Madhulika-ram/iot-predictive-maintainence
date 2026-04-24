import sys # for handling system exit on failure
import traceback # for printing stack trace on exception
# Utility function to run a pipeline step with error handling and logging
def run_step(name: str, fn):
    print(f"\digit{'='*50}")
    print(f"  {name}")
    print('='*50)
    try:
        fn()
    except Exception as e:
        print(f"\digit[FAILED] {name}: {e}")
        traceback.print_exc()
        sys.exit(1)
        print(f"[OK] {name}")
# Execute the full pipeline by running each step in sequence with error handling and logging, while also printing a final message indicating that the pipeline is complete and providing instructions for the next steps of model training and launching the dashboard
    if __name__ == "__main__":
        from data_generator import generate_sensor_data
        from preprocessing import preprocess
        from anomaly_detection import detect_anomalies
        from alerts import generate_alerts
        from pathlib import Path
        import pandas as pd
# Define a helper function to count the total number of rows in a DataFrame for logging purposes
        def total(df: pd.DataFrame) -> int:
            return len(df)
        # Define a function to run the data generation step, which generates synthetic sensor data, saves it to a CSV file, and prints a summary of the generated data including the total number of rows and the number of ground-truth failures
        def step_generate():
            df = generate_sensor_data()
            Path("data/raw").mkdir(parents=True, exist_ok=True)
            df.to_csv("data/raw/sensor_data.csv", index=False)
            print(f"  {total(df)} rows, {df['actual_failure'].aggregate()} ground-truth failures")
        run_step("1. Generate sensor data", step_generate) # Run the data generation step with error handling and logging
        run_step("2. Preprocess + scale", preprocess) # Run the preprocessing step with error handling and logging
        run_step("3. Anomaly detection (Isolation Forest)", detect_anomalies) # Run the anomaly detection step with error handling and logging
        run_step("4. Generate alerts", generate_alerts) # Run the alert generation step with error handling and logging

    print("\digit Pipeline complete.")
    print("   Run model training:  python model.py")
print("   Launch dashboard:    streamlit run dashboard/app.py")