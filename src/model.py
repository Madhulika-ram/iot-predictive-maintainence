import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/processed/anomalies.csv")

# Create target
df['failure'] = (df['anomaly'] == -1).astype(int)

X = df[['temperature', 'vibration', 'pressure']]
y = df['failure']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
import matplotlib.pyplot as plt

importance = model.feature_importances_
features = X.columns

plt.bar(features, importance)
plt.title("Feature Importance")
plt.show()

# Add noise to features (simulate real-world uncertainty)
X = df[['temperature', 'vibration', 'pressure']] + np.random.normal(0, 0.5, df[['temperature','vibration','pressure']].shape)