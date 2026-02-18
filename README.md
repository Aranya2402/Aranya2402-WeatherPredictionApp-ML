# Backend
cd backend
pip install -r requirements.txt
python data_preprocessing.py
python train.py
uvicorn main:app --reload --port 8000

# Frontend (in another terminal)
cd frontend
pip install -r requirements.txt
python app.py

# Heavy Rainfall Prediction in Sri Lanka using XGBoost

## 📋 Project Overview
This project predicts heavy rainfall (>200mm per week) in Sri Lanka using XGBoost, with a complete web application interface.

## 🎯 Features
- Predicts heavy rainfall using weather parameters
- XGBoost model with strong regularization
- SHAP explainability for model interpretation
- FastAPI backend
- Flask web interface
- Docker support for easy deployment

## 📊 Dataset
- Source: Kaggle weather dataset for Sri Lanka (2010-2023)
- Original: 140,000+ daily records
- Processed: 703 weekly records
- Features: Temperature, precipitation, wind speed, weather codes
- Target: Heavy rain (>200mm/week)


📈 Model Performance
Accuracy: ~0.95-0.98

F1 Score: ~0.97-0.99

ROC-AUC: ~0.98-1.00

🔧 Technologies Used
XGBoost for machine learning

SHAP for model explainability

FastAPI for backend API

Flask for frontend

Docker for containerization

Python, Pandas, Scikit-learn