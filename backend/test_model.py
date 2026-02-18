# backend/test_model.py
import joblib
import pandas as pd

# Load the model
model_data = joblib.load('xgboost_model.pkl')
model = model_data['model']
feature_names = model_data['feature_names']

print("Model expects these features:")
print(feature_names)
print(f"Number of features: {len(feature_names)}")