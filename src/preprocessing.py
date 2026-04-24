import pandas as pd 
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path
# Preprocess the raw sensor data by scaling the features and saving the scaler for future use
def preprocess(input_path: str = "data/raw/sensor_data.csv",  
               output_dir: str = "data/processed") -> pd.DataFrame:

    output_path = Path(output_dir) # create a Path object for the output directory
    output_path.mkdir(parents=True, exist_ok=True) # ensure the output directory exists

    df = pd.read_csv(input_path)

    feature_cols = ['temperature', 'vibration', 'pressure'] # specify the columns to be scaled
    features = df[feature_cols] # extract the features to be scaled from the DataFrame

    scaler = StandardScaler() # initialize the StandardScaler to standardize the features by removing the mean and scaling to unit variance
    scaled = scaler.fit_transform(features) # fit the scaler to the features and transform them to have mean 0 and variance 1

    joblib.dump(scaler, output_path / "scaler.pkl") # save the fitted scaler to a file for later use in scaling new data during inference

    scaled_df = pd.DataFrame(scaled, columns=feature_cols) # create a new DataFrame to hold the scaled features with the same column names as the original features

    scaled_df.insert(0, "timestamp", df["timestamp"].values) # insert the timestamp column back into the scaled DataFrame as the first column
    if "actual_failure" in df.columns: # if the original DataFrame contains the actual_failure column, add it back to the scaled DataFrame without scaling since it's a binary label
        scaled_df["actual_failure"] = df["actual_failure"].values # add the actual_failure column back to the scaled DataFrame without scaling since it's a binary label

    scaled_df.to_csv(output_path / "scaled_data.csv", index=False) # save the scaled DataFrame to a new CSV file in the output directory without the index
    print(f"Preprocessing done — scaler saved to {output_path / 'scaler.pkl'}") # print a message indicating that preprocessing is complete and the scaler has been saved to the specified path
    return scaled_df

if __name__  == "__main__": # execute the preprocess function when the script is run directly
    preprocess()