import numpy as np
import pandas as pd
import joblib 
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Train a Random Forest model on the preprocessed data with ground-truth labels, evaluate its performance, save the trained model for later use in the dashboard, and generate a feature importance plot to visualize which features are most influential in predicting failures
def train_model(input_path: str = "data/processed/anomalies.csv",
                model_output: str = "data/processed/rf_model.pkl") -> None:

    Path(model_output).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    # If actual_failure column is missing, it likely means the data was not generated with labels. In that case, we cannot train a supervised model, so we raise an error prompting the user to re-run the data generation step to include ground-truth labels.
    if "actual_failure" not in df.columns:
        raise ValueError(
            "Column 'actual_failure' not found. Re-run data_generator.py to get ground-truth labels."
        )
# Define the feature columns and target variable for model training, print the class distribution to understand the imbalance in the dataset, split the data into training and testing sets while stratifying to maintain class distribution, train a Random Forest classifier with class weighting to handle the imbalance, evaluate the model's performance using accuracy and classification report, save the trained model to a file for later use in the dashboard, and generate a feature importance plot to visualize which features are most influential in predicting failures
    feature_cols = ['temperature', 'vibration', 'pressure']
    X = df[feature_cols]
    y = df['actual_failure']

    print(f"Class distribution — Normal: {(y==0).sum()}, Failure: {(y==1).sum()}")
# Split the data into training and testing sets with stratification to maintain class distribution, using 20% of the data for testing and a random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
# Train a Random Forest classifier with 100 trees and class weighting to handle the imbalanced dataset (950 normal : 50 failure), and fit the model to the training data
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',  # handles the imbalanced dataset (950 normal : 50 failure)
        random_state=42
    )
    model.fit(X_train, y_train)
# Predict on the test set and evaluate the model's performance using accuracy and a classification report that includes precision, recall, and F1-score for both classes (normal and failure)
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))

    # Save trained model so it can be loaded in the dashboard for live scoring
    joblib.dump(model, model_output)
    print(f"Model saved to {model_output}")

    # Feature importance plot
    importance = model.feature_importances_
    plt.figure(figsize=(6, 4))
    plt.bar(feature_cols, importance, color=['#378ADD', '#1D9E75', '#D85A30'])
    plt.title("Feature Importance")
    plt.ylabel("Importance score")
    plt.tight_layout()
    plt.savefig("data/processed/feature_importance.png", dpi=150)
    plt.show()
    print("Feature importance chart saved.")

# Execute the model training function when the script is run directly
if __name__ == "__main__":
    train_model()
