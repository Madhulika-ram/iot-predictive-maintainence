import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # for classification
from sklearn.model_selection import train_test_split # for splitting data into training and testing sets
from sklearn.metrics import accuracy_score, classification_report # for evaluating model performance

df = pd.read_csv("data/processed/anomalies.csv")

# Create target
df['failure'] = (df['anomaly'] == -1).astype(int) # create binary target variable where 1 indicates failure (anomaly) and 0 indicates normal operation

X = df[['temperature', 'vibration', 'pressure']] # features for model training
y = df['failure'] # target variable for model training

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100) # initialize Random Forest model with 100 trees
model.fit(X_train, y_train) # fit the model to the training data

y_pred = model.predict(X_test) # predict on the test set and evaluate performance

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
import matplotlib.pyplot as plt

importance = model.feature_importances_ # get feature importance scores from the trained model
features = X.columns # plot feature importance to visualize which features are most influential in predicting failures

plt.bar(features, importance)
plt.title("Feature Importance")
plt.show()

# add small random noise to the features to make the model more robust to real-world data variability
X = df[['temperature', 'vibration', 'pressure']] + np.random.normal(0, 0.5, df[['temperature','vibration','pressure']].shape) 
