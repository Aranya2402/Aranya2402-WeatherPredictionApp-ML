"""
Find the best threshold for class balance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb

# Load data
df = pd.read_csv('data/weekly_weather_data.csv')

# Feature engineering (same as before)
df['time'] = pd.to_datetime(df['time'])
df['month'] = df['time'].dt.month
df['year'] = df['time'].dt.year
df = df.drop('time', axis=1)

df['country'] = pd.Categorical(df['country']).codes
df['city'] = pd.Categorical(df['city']).codes

df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
df['rain_per_windspeed'] = df['rain_sum'] / (df['windspeed_10m_max'] + 0.001)

feature_cols = ['weathercode', 'temperature_2m_max', 'temperature_2m_min', 
                'precipitation_sum', 'windspeed_10m_max', 'country', 'city',
                'month', 'year', 'temp_range', 'rain_per_windspeed']

X = df[feature_cols]

print("=" * 60)
print("FINDING BEST THRESHOLD")
print("=" * 60)

thresholds = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

results = []

for threshold in thresholds:
    # Create target
    y = np.where(df['rain_sum'] > threshold, 1, 0)
    
    # Skip if too few samples
    if sum(y==0) < 5 or sum(y==1) < 5:
        print(f"\nThreshold {threshold}mm: Skipped (too few samples)")
        continue
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Calculate scale_pos_weight
    scale_pos_weight = sum(y_train==0) / sum(y_train==1)
    
    # Train
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({
        'threshold': threshold,
        'class0': sum(y==0),
        'class1': sum(y==1),
        'f1_score': f1,
        'scale_pos_weight': scale_pos_weight
    })
    
    print(f"\nThreshold {threshold}mm:")
    print(f"   Class 0: {sum(y==0)}, Class 1: {sum(y==1)}")
    print(f"   F1 Score: {f1:.4f}")

# Show best threshold
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(results)
print(results_df.sort_values('f1_score', ascending=False).to_string(index=False))

best = results_df.loc[results_df['f1_score'].idxmax()]
print(f"\n✅ Best threshold: {best['threshold']}mm (F1: {best['f1_score']:.4f})")