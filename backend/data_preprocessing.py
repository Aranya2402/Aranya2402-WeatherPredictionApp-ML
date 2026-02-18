"""
Data Preprocessing for Weather Prediction
This script cleans and prepares the weekly weather data for model training
Based on the provided cleaning code
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime

class WeatherDataPreprocessor:
    def __init__(self, data_path):
        """
        Initialize the preprocessor with data path
        """
        self.data_path = data_path
        self.df = None
        self.encoders = {}
        
    def load_and_clean_data(self):
        """
        Load the weekly weather data and perform initial cleaning
        This follows the provided cleaning code
        """
        print("=" * 60)
        print("STEP 1: Loading and Initial Cleaning")
        print("=" * 60)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Original dataset shape: {self.df.shape}")
        print(f"Original columns: {list(self.df.columns)}")
        
        # Data Quality Report
        print("\n--- Data Quality Report ---")
        
        # 1. Missing Values
        print("\n1. Missing Values per Column:")
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found.")
        
        # 2. Data Types
        print("\n2. Data Types:")
        print(self.df.dtypes)
        
        # 3. Duplicate Rows
        duplicates = self.df.duplicated().sum()
        print(f"\n3. Duplicate Rows found: {duplicates}")
        
        # 4. Statistical Summary
        print("\n4. Statistical Summary (Check for Outliers):")
        print(self.df.describe())
        
        # 5. Time Intervals Check
        if 'time' in self.df.columns:
            self.df['time'] = pd.to_datetime(self.df['time'])
            time_diff = self.df['time'].diff().value_counts()
            print("\n5. Time Intervals (Should be consistent):")
            print(time_diff)
        
        return self.df
    
    def create_features(self):
        """
        Create additional features from existing data
        """
        print("\n" + "=" * 60)
        print("STEP 2: Feature Engineering")
        print("=" * 60)
        
        # Extract time-based features
        if 'time' in self.df.columns:
            self.df['year'] = self.df['time'].dt.year
            self.df['month'] = self.df['time'].dt.month
            self.df['week'] = self.df['time'].dt.isocalendar().week
            self.df['day_of_year'] = self.df['time'].dt.dayofyear
            
            print("Added time-based features:")
            print(f"  - year: {self.df['year'].nunique()} unique years")
            print(f"  - month: {self.df['month'].nunique()} months")
            print(f"  - week: {self.df['week'].nunique()} weeks")
        
        # Create temperature range feature
        if 'temperature_2m_max' in self.df.columns and 'temperature_2m_min' in self.df.columns:
            self.df['temperature_range'] = self.df['temperature_2m_max'] - self.df['temperature_2m_min']
            print("Added temperature_range feature")
        
        # Create rain intensity feature
        if 'rain_sum' in self.df.columns:
            # Categorize rain intensity
            self.df['rain_intensity'] = pd.cut(
                self.df['rain_sum'],
                bins=[-1, 0, 10, 50, float('inf')],
                labels=['none', 'light', 'moderate', 'heavy']
            )
            print("Added rain_intensity feature")
            print(f"  Rain intensity distribution:")
            print(f"    {self.df['rain_intensity'].value_counts().to_dict()}")
        
        return self.df
    
    def create_target_variable(self):
        """
        Create target variable RainTomorrow
        """
        print("\n" + "=" * 60)
        print("STEP 3: Creating Target Variable")
        print("=" * 60)
        
        # Create target: 1 if rain_sum > 10mm (adjustable threshold)
        threshold = 10.0
        self.df['RainTomorrow'] = np.where(self.df['rain_sum'] > threshold, 1, 0)
        
        print(f"Threshold used: {threshold}mm")
        print(f"Class distribution:")
        print(f"  No Rain (0): {(self.df['RainTomorrow'] == 0).sum()} ({((self.df['RainTomorrow'] == 0).sum()/len(self.df))*100:.1f}%)")
        print(f"  Rain (1): {(self.df['RainTomorrow'] == 1).sum()} ({((self.df['RainTomorrow'] == 1).sum()/len(self.df))*100:.1f}%)")
        
        return self.df
    
    def select_features(self):
        """
        Select relevant features for the model
        """
        print("\n" + "=" * 60)
        print("STEP 4: Feature Selection")
        print("=" * 60)
        
        # Features to keep (based on domain knowledge)
        features_to_keep = [
            'weathercode',
            'temperature_2m_max',
            'temperature_2m_min',
            'precipitation_sum',
            'rain_sum',
            'windspeed_10m_max',
            'country',
            'city',
            'year',
            'month',
            'temperature_range',
            'rain_intensity'
        ]
        
        # Keep only available features
        available_features = [f for f in features_to_keep if f in self.df.columns]
        self.feature_columns = available_features.copy()
        
        # Keep selected features plus target
        self.df = self.df[available_features + ['RainTomorrow']].copy()
        
        print(f"Selected features: {available_features}")
        print(f"Number of features: {len(available_features)}")
        print(f"New shape: {self.df.shape}")
        
        return self.df
    
    def handle_categorical_variables(self):
        """
        Encode categorical variables
        """
        print("\n" + "=" * 60)
        print("STEP 5: Encoding Categorical Variables")
        print("=" * 60)
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != 'RainTomorrow']
        
        print(f"Categorical columns to encode: {list(categorical_cols)}")
        
        for col in categorical_cols:
            print(f"\nEncoding: {col}")
            print(f"  Unique values before: {self.df[col].nunique()}")
            if self.df[col].nunique() < 10:
                print(f"  Values: {dict(self.df[col].value_counts())}")
            
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.encoders[col] = le
            
            print(f"  Unique values after: {self.df[col].nunique()}")
        
        return self.df, self.encoders
    
    def check_class_balance(self):
        """
        Check and report class balance
        """
        print("\n" + "=" * 60)
        print("STEP 6: Class Balance Check")
        print("=" * 60)
        
        class_counts = self.df['RainTomorrow'].value_counts()
        class_percentages = self.df['RainTomorrow'].value_counts(normalize=True) * 100
        
        print(f"Class 0 (No Rain): {class_counts[0]} ({class_percentages[0]:.1f}%)")
        print(f"Class 1 (Rain): {class_counts[1]} ({class_percentages[1]:.1f}%)")
        
        if class_percentages[0] < 40 or class_percentages[1] < 40:
            print("\n⚠️  Warning: Dataset is imbalanced!")
            print("   Consider using class weights or resampling techniques.")
        else:
            print("\n✅ Dataset is reasonably balanced.")
    
    def save_processed_data(self, output_path='processed_weather_data.csv'):
        """
        Save the processed dataset
        """
        print("\n" + "=" * 60)
        print("STEP 7: Saving Processed Data")
        print("=" * 60)
        
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
        print(f"Final shape: {self.df.shape}")
        print(f"Final columns: {list(self.df.columns)}")
        
        # Save encoders
        joblib.dump(self.encoders, 'encoders.pkl')
        print(f"Encoders saved to: encoders.pkl")
        
        # Save feature list
        with open('feature_list.txt', 'w') as f:
            f.write('\n'.join(self.feature_columns))
        print(f"Feature list saved to: feature_list.txt")
    
    def run_pipeline(self):
        """
        Run the complete preprocessing pipeline
        """
        print("\n" + "=" * 60)
        print("WEATHER DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        self.load_and_clean_data()
        self.create_features()
        self.create_target_variable()
        self.select_features()
        self.handle_categorical_variables()
        self.check_class_balance()
        self.save_processed_data()
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        return self.df, self.encoders

# Run the pipeline if script is executed directly
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = WeatherDataPreprocessor('data/weekly_weather_data.csv')
    
    # Run the pipeline
    processed_df, encoders = preprocessor.run_pipeline()
    
    print(f"\n✅ Final dataset ready for training!")
    print(f"   Features: {len(processed_df.columns) - 1}")
    print(f"   Samples: {len(processed_df)}")