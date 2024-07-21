Defect Prediction Model

Overview

This project aims to build a defect prediction model using data from Jira tickets. The model is trained using a Random Forest classifier to predict the likelihood of a defect based on various features from the dataset.
Project Structure

	•	ModelDevelopment.py: Main script to preprocess the data, train the model, evaluate its performance, and save the trained model and encoders.
    •   NewDataPrediction.py : To Predict the newdat.csv using the  developed model 
	•	DataPreprocessing.py: Module containing the preprocess_data function to preprocess the Jira data.
	•	jira_data.csv: CSV file containing the raw Jira data.
	•	newdat.csv: CSV file containing new Jira data for prediction.
	•	priority_encoder.pkl, status_encoder.pkl, component_encoder.pkl, scaler.pkl, defect_prediction_model.pkl: Saved model and encoders.
	•	Dockerfile: Docker configuration file.
	•	requirements.txt: List of dependencies to be installed in the Docker container.
	•	entrypoint.sh: Entrypoint script to run the prediction script inside the Docker container.