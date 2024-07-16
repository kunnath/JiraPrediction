# DataPreprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data['resolved_date'].fillna('2024-07-03', inplace=True)

    # Convert date columns to datetime
    data['created_date'] = pd.to_datetime(data['created_date'])
    data['resolved_date'] = pd.to_datetime(data['resolved_date'])

    # Encode categorical variables
    priority_encoder = LabelEncoder()
    status_encoder = LabelEncoder()
    component_encoder = LabelEncoder()
    
    data['priority'] = priority_encoder.fit_transform(data['priority'])
    data['status'] = status_encoder.fit_transform(data['status'])
    data['component'] = component_encoder.fit_transform(data['component'])

    # Feature engineering
    data['resolution_time'] = (data['resolved_date'] - data['created_date']).dt.days
    data.drop(['jira_id', 'requirement', 'created_date', 'resolved_date'], axis=1, inplace=True)

    # Split the data into features and labels
    X = data.drop('defect', axis=1)
    y = data['defect']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, priority_encoder, status_encoder, component_encoder