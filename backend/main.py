"""
FastAPI Backend for Weather Prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Weather Prediction API",
    description="API for predicting heavy rain using XGBoost",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model_path = os.path.join(os.path.dirname(__file__), "xgboost_model.pkl")

try:
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    best_params = model_data.get('best_params', {})
    
    print(f"✅ Model loaded successfully!")
    print(f"   Model type: XGBoost Classifier")
    print(f"   Features: {len(feature_names)}")
    print(f"   Best params: {best_params}")
    print(f"   Features: {feature_names}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    feature_names = []

# Define input model (based on original data, not engineered features)
class WeatherInput(BaseModel):
    weathercode: float = Field(..., description="Weather code (WMO code)", ge=0, le=100)
    temperature_2m_max: float = Field(..., description="Maximum temperature in °C", ge=-50, le=50)
    temperature_2m_min: float = Field(..., description="Minimum temperature in °C", ge=-50, le=50)
    temperature_2m_mean: float = Field(..., description="Mean temperature in °C", ge=-50, le=50)
    precipitation_sum: float = Field(..., description="Total precipitation in mm", ge=0)
    rain_sum: float = Field(..., description="Total rain in mm", ge=0)
    windspeed_10m_max: float = Field(..., description="Maximum wind speed in km/h", ge=0, le=200)
    country: str = Field(..., description="Country name")
    city: str = Field(..., description="City name")
    
    class Config:
        schema_extra = {
            "example": {
                "weathercode": 63.0,
                "temperature_2m_max": 28.9,
                "temperature_2m_min": 22.9,
                "temperature_2m_mean": 25.9,
                "precipitation_sum": 45.0,
                "rain_sum": 45.0,
                "windspeed_10m_max": 14.7,
                "country": "Sri Lanka",
                "city": "Colombo"
            }
        }

class PredictionResponse(BaseModel):
    success: bool
    prediction: int
    probability: float
    heavy_rain_tomorrow: str
    confidence: str
    message: str

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "app_name": "Heavy Rain Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_type": "XGBoost Classifier",
        "threshold": "200mm (very heavy rain)",
        "endpoints": {
            "/": "GET - This information",
            "/predict": "POST - Make a prediction",
            "/features": "GET - List required features",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "features_count": len(feature_names) if feature_names else 0
    }

@app.get("/features")
def get_features():
    """Get list of required features"""
    return {
        "features": feature_names,
        "count": len(feature_names),
        "description": "Features required for heavy rain prediction",
        "example": {
            "weathercode": 63.0,
            "temperature_2m_max": 28.9,
            "temperature_2m_min": 22.9,
            "temperature_2m_mean": 25.9,
            "precipitation_sum": 45.0,
            "rain_sum": 45.0,
            "windspeed_10m_max": 14.7,
            "country": "Sri Lanka",
            "city": "Colombo"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: WeatherInput):
    """
    Make a prediction for heavy rain tomorrow (>200mm)
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert input to dictionary
        input_dict = data.dict()
        print(f"Received input: {input_dict}")
        
        # Create base DataFrame
        df = pd.DataFrame([input_dict])
        
        # Add time features
        now = datetime.now()
        df['year'] = now.year
        df['month'] = now.month
        df['day'] = now.day
        df['day_of_week'] = now.weekday()
        df['week_of_year'] = now.isocalendar()[1]
        df['quarter'] = (now.month - 1) // 3 + 1
        
        # Encode categorical variables
        df['country'] = 0  # Sri Lanka is always 0
        df['city'] = 0 if input_dict['city'].lower() == 'colombo' else 1
        
        # Create engineered features (matching training)
        df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
        df['temp_squared'] = df['temperature_2m_mean'] ** 2
        df['temp_cubed'] = df['temperature_2m_mean'] ** 3
        df['wind_squared'] = df['windspeed_10m_max'] ** 2
        df['wind_log'] = np.log1p(df['windspeed_10m_max'])
        
        # Weather category
        weathercode = df['weathercode'].values[0]
        if weathercode <= 19:
            weather_category = 0
        elif weathercode <= 29:
            weather_category = 1
        elif weathercode <= 39:
            weather_category = 2
        elif weathercode <= 49:
            weather_category = 3
        elif weathercode <= 69:
            weather_category = 4
        elif weathercode <= 79:
            weather_category = 5
        else:
            weather_category = 6
        df['weather_category'] = weather_category
        
        # Interaction features
        df['temp_wind_interaction'] = df['temperature_2m_mean'] * df['windspeed_10m_max']
        
        # Rolling averages (use current values as proxy)
        df['temp_rolling_avg'] = df['temperature_2m_mean']
        
        # Drop original highly correlated features (same as training)
        df = df.drop(['rain_sum', 'precipitation_sum'], axis=1, errors='ignore')
        
        # Ensure all features are present in correct order
        missing_cols = set(feature_names) - set(df.columns)
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}. Adding with default 0.")
            for col in missing_cols:
                df[col] = 0
        
        # Select only the features the model expects, in the right order
        X_pred = df[feature_names].astype(float)
        
        print(f"Prediction features: {X_pred.columns.tolist()}")
        
        # Make prediction
        prediction = model.predict(X_pred)[0]
        probability = model.predict_proba(X_pred)[0][1]
        
        return {
            "success": True,
            "prediction": int(prediction),
            "probability": float(probability),
            "heavy_rain_tomorrow": "Yes" if prediction == 1 else "No",
            "confidence": f"{probability*100:.1f}%",
            "message": f"{'Heavy rain' if prediction == 1 else 'No heavy rain'} expected tomorrow with {probability*100:.1f}% confidence"
        }
        
    except Exception as e:
        import traceback
        print("=" * 60)
        print("ERROR IN PREDICTION:")
        traceback.print_exc()
        print("=" * 60)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)