import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn
import numpy as np

# Check versions to ensure correct environment
print(f'sklearn: {sklearn.__version__}')
print(f'pandas: {pd.__version__}')
print(f'numpy: {np.__version__}')
print(f'joblib: {joblib.__version__}')

# Load the new data
new_data = pd.read_csv('newdat.csv')

# Convert date columns to datetime
new_data['created_date'] = pd.to_datetime(new_data['created_date'])
new_data['resolved_date'] = pd.to_datetime(new_data['resolved_date'])

# Handle missing values in 'resolved_date' if any
if new_data['resolved_date'].isna().sum() > 0:
    new_data['resolved_date'].fillna(pd.Timestamp('2099-12-31'), inplace=True)

# Load the encoders and scaler
priority_encoder = joblib.load('./priority_encoder.pkl')
status_encoder = joblib.load('./status_encoder.pkl')
component_encoder = joblib.load('./component_encoder.pkl')
scaler = joblib.load('./scaler.pkl')

# Encode categorical variables
new_data['priority'] = priority_encoder.transform(new_data['priority'])
new_data['status'] = status_encoder.transform(new_data['status'])
new_data['component'] = component_encoder.transform(new_data['component'])

# Feature engineering
new_data['resolution_time'] = (new_data['resolved_date'] - new_data['created_date']).dt.days

# Drop columns not used in model
new_data.drop(['jira_id', 'requirement', 'created_date', 'resolved_date'], axis=1, inplace=True)

# Ensure the order of columns matches what was used during training
expected_columns = ['priority', 'status', 'component', 'resolution_time']
new_data = new_data[expected_columns]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
new_data_imputed = imputer.fit_transform(new_data)

# Normalize the features
new_data_scaled = scaler.transform(new_data_imputed)

# Load the model and make predictions
loaded_model = joblib.load('./defect_prediction_model.pkl')
new_predictions = loaded_model.predict(new_data_scaled)

# Convert scaled data back to DataFrame to merge with predictions
new_data_df = pd.DataFrame(new_data_imputed, columns=expected_columns)
new_data_df['defect_prediction'] = new_predictions

print(new_data_df)