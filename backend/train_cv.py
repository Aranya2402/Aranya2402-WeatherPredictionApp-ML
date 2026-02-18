"""
Analyze model for overfitting
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve  # Correct import
from sklearn.model_selection import train_test_split

# Load model and data
model_data = joblib.load('xgboost_model_cv.pkl')
model = model_data['model']
feature_names = model_data['feature_names']

# Load and prepare data (same as in training)
df = pd.read_csv('data/weekly_weather_data.csv')

# Prepare features (copy preprocessing from train_cv.py)
df['time'] = pd.to_datetime(df['time'])
df['month'] = df['time'].dt.month
df['year'] = df['time'].dt.year
df['day_of_year'] = df['time'].dt.dayofyear
df = df.drop('time', axis=1)

df['country'] = pd.Categorical(df['country']).codes
df['city'] = pd.Categorical(df['city']).codes

df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
df['rain_per_windspeed'] = df['rain_sum'] / (df['windspeed_10m_max'] + 0.001)
df['precip_ratio'] = df['precipitation_sum'] / (df['rain_sum'] + 0.001)

# Create target
threshold = 50.0
df['HeavyRain'] = np.where(df['rain_sum'] > threshold, 1, 0)

X = df[feature_names]
y = df['HeavyRain']

print("=" * 60)
print("OVERFITTING ANALYSIS")
print("=" * 60)

# 1. Learning Curve
print("\n📈 Generating Learning Curve...")
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Cross-Validation Score')
plt.xlabel('Training Examples')
plt.ylabel('F1 Score')
plt.title('Learning Curve - Check for Overfitting')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=300)
plt.show()

# 2. Feature Importance Distribution
importance = model.feature_importances_
plt.figure(figsize=(12, 6))
plt.bar(range(len(importance)), sorted(importance, reverse=True))
plt.xlabel('Feature Rank')
plt.ylabel('Importance Score')
plt.title('Feature Importance Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('importance_distribution.png', dpi=300)
plt.show()

print("\n🔝 Feature Importance Distribution:")
for i, imp in enumerate(sorted(importance, reverse=True)[:5]):
    print(f"   Top {i+1}: {imp:.4f}")
print(f"   Top feature accounts for {sorted(importance, reverse=True)[0]*100:.1f}% of importance")

# 3. Prediction confidence analysis
y_pred_proba = model.predict_proba(X)[:, 1]
print("\n🎯 Prediction Confidence:")
print(f"   Mean confidence for class 0: {y_pred_proba[y==0].mean():.4f}")
print(f"   Mean confidence for class 1: {y_pred_proba[y==1].mean():.4f}")
print(f"   Confidence std dev: {y_pred_proba.std():.4f}")

# 4. Check for perfect separation
if len(y_pred_proba[y==0]) > 0 and len(y_pred_proba[y==1]) > 0:
    if (y_pred_proba[y==0].max() < y_pred_proba[y==1].min()):
        print("\n⚠️  WARNING: Perfect class separation detected - possible overfitting!")
    else:
        print("\n✅ No perfect separation - good sign!")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)