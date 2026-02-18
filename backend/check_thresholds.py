# check_thresholds.py
import pandas as pd
import numpy as np

df = pd.read_csv('data/weekly_weather_data.csv')

print("=" * 60)
print("CHECKING DIFFERENT THRESHOLDS")
print("=" * 60)

thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

for threshold in thresholds:
    df['RainTomorrow'] = np.where(df['rain_sum'] > threshold, 1, 0)
    class_counts = df['RainTomorrow'].value_counts()
    print(f"\nThreshold: {threshold}mm")
    print(f"  Class 0 (no rain): {class_counts.get(0, 0)} samples")
    print(f"  Class 1 (rain): {class_counts.get(1, 0)} samples")
    print(f"  Ratio: {class_counts.get(0, 0)/len(df)*100:.1f}% / {class_counts.get(1, 0)/len(df)*100:.1f}%")