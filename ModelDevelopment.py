# main_script.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import DataPreprocessing

# Run data preprocessing
file_path = 'jira_data.csv'
X, y, scaler, priority_encoder, status_encoder, component_encoder = DataPreprocessing.preprocess_data(file_path)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the model and encoders
joblib.dump(model, 'defect_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(priority_encoder, 'priority_encoder.pkl')
joblib.dump(status_encoder, 'status_encoder.pkl')
joblib.dump(component_encoder, 'component_encoder.pkl')