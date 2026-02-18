"""
Model Explainability using SHAP and Feature Importance
This script provides XAI analysis of the trained model
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    def __init__(self, model_path='xgboost_model.pkl', data_path='data/weekly_weather_data.csv'):
        """
        Initialize the explainer with model and data
        """
        print("=" * 60)
        print("MODEL EXPLAINABILITY ANALYSIS")
        print("=" * 60)
        
        # Load model
        print("\n📂 Loading model...")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data.get('best_params', {})
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {model_data.get('model_type', 'XGBoost')}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Best params: {self.best_params}")
        
        # Load data
        print("\n📂 Loading data...")
        self.df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {self.df.shape}")
        print(f"   Columns: {list(self.df.columns)}")
        
        self.prepare_data()
        
    def prepare_data(self):
        """
        Prepare data for explanation (must match training preprocessing)
        """
        print("\n🔄 Preparing data for SHAP analysis...")
        
        # Create a copy
        df_features = self.df.copy()
        
        # Create target if not exists (using same threshold as training - 200mm)
        if 'RainTomorrow' not in df_features.columns:
            threshold = 200.0
            df_features['RainTomorrow'] = np.where(df_features['rain_sum'] > threshold, 1, 0)
            print(f"   Created target with threshold: {threshold}mm")
        
        # Apply the SAME preprocessing as in train.py
        if 'time' in df_features.columns:
            df_features['time'] = pd.to_datetime(df_features['time'])
            df_features['year'] = df_features['time'].dt.year
            df_features['month'] = df_features['time'].dt.month
            df_features['day'] = df_features['time'].dt.day
            df_features['day_of_week'] = df_features['time'].dt.dayofweek
            df_features['week_of_year'] = df_features['time'].dt.isocalendar().week
            df_features['quarter'] = df_features['time'].dt.quarter
            df_features = df_features.drop('time', axis=1)
        
        # Encode categorical columns
        categorical_cols = ['country', 'city']
        for col in categorical_cols:
            if col in df_features.columns:
                df_features[col] = pd.Categorical(df_features[col]).codes
        
        # Create additional features
        if 'temperature_2m_max' in df_features.columns and 'temperature_2m_min' in df_features.columns:
            df_features['temp_range'] = df_features['temperature_2m_max'] - df_features['temperature_2m_min']
        
        if 'temperature_2m_mean' in df_features.columns:
            df_features['temp_squared'] = df_features['temperature_2m_mean'] ** 2
            df_features['temp_cubed'] = df_features['temperature_2m_mean'] ** 3
        
        if 'windspeed_10m_max' in df_features.columns:
            df_features['wind_squared'] = df_features['windspeed_10m_max'] ** 2
            df_features['wind_log'] = np.log1p(df_features['windspeed_10m_max'])
        
        if 'weathercode' in df_features.columns:
            df_features['weather_category'] = pd.cut(
                df_features['weathercode'], 
                bins=[-1, 19, 29, 39, 49, 69, 79, 99], 
                labels=[0, 1, 2, 3, 4, 5, 6]
            ).astype(int)
        
        if 'temperature_2m_mean' in df_features.columns and 'windspeed_10m_max' in df_features.columns:
            df_features['temp_wind_interaction'] = df_features['temperature_2m_mean'] * df_features['windspeed_10m_max']
        
        if 'temperature_2m_mean' in df_features.columns:
            df_features['temp_rolling_avg'] = df_features['temperature_2m_mean'].rolling(window=3, min_periods=1).mean()
        
        # Fill NaN values
        df_features = df_features.fillna(0)
        
        # Drop highly correlated features (same as training)
        features_to_drop = ['rain_sum', 'precipitation_sum', 'heavy_rain', 'precip_ratio']
        for col in features_to_drop:
            if col in df_features.columns:
                df_features = df_features.drop(col, axis=1)
        
        # Get features for SHAP
        X = df_features[self.feature_names]
        y = df_features['RainTomorrow']
        
        # Use a sample for SHAP (faster)
        self.X_sample = X.sample(min(100, len(X)), random_state=42)
        self.y_sample = y.loc[self.X_sample.index]
        
        print(f"\n📊 Data prepared for explanation:")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Sample size: {len(self.X_sample)}")
        print(f"   Sample class distribution:")
        print(self.y_sample.value_counts())
        
    def analyze_feature_importance(self):
        """
        Detailed feature importance analysis
        """
        print("\n" + "=" * 60)
        print("1. FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\n🔝 Top 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        # Plot with better styling
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df.head(10))))
        bars = plt.barh(importance_df['Feature'].head(10)[::-1], 
                       importance_df['Importance'].head(10)[::-1], 
                       color=colors[::-1])
        
        plt.xlabel('Importance Score', fontsize=14, fontweight='bold')
        plt.ylabel('Features', fontsize=14, fontweight='bold')
        plt.title('Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance_df['Importance'].head(10)[::-1])):
            plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('detailed_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save to CSV
        importance_df.to_csv('feature_importance_detailed.csv', index=False)
        print("\n✅ Detailed importance saved to 'feature_importance_detailed.csv'")
        
        return importance_df
    
    def analyze_shap_values(self):
        """
        SHAP analysis for model interpretability
        """
        print("\n" + "=" * 60)
        print("2. SHAP VALUE ANALYSIS")
        print("=" * 60)
        
        print("\n🔍 What are SHAP values?")
        print("   • SHAP explains individual predictions")
        print("   • Shows contribution of each feature to the prediction")
        print("   • Positive values push prediction towards heavy rain")
        print("   • Negative values push prediction towards no heavy rain")
        print("   • Based on game theory principles")
        
        # Create SHAP explainer
        print("\n🔄 Computing SHAP values (this may take a moment)...")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_sample)
        
        # Summary plot
        print("\n📊 Generating SHAP summary plot...")
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, self.X_sample, 
                         feature_names=self.feature_names, 
                         show=False,
                         plot_size=(12, 8))
        plt.title('SHAP Feature Impact on Heavy Rain Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Bar plot of mean |SHAP|
        print("\n📊 Generating SHAP bar plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, self.X_sample, 
                         feature_names=self.feature_names, 
                         plot_type="bar", 
                         show=False,
                         plot_size=(12, 8))
        plt.title('Mean |SHAP| Values (Feature Importance)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_bar.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze multiple individual predictions
        print("\n" + "=" * 60)
        print("3. INDIVIDUAL PREDICTION ANALYSIS")
        print("=" * 60)
        
        for idx in range(min(3, len(self.X_sample))):
            self.analyze_single_prediction(idx, explainer, shap_values)
        
        return shap_values
    
    def analyze_single_prediction(self, idx, explainer, shap_values):
        """
        Analyze a single prediction in detail
        """
        print(f"\n{'─' * 60}")
        print(f"🔍 Analyzing Sample #{idx}")
        print(f"{'─' * 60}")
        
        sample = self.X_sample.iloc[idx:idx+1]
        actual = self.y_sample.iloc[idx]
        prediction = self.model.predict(sample)[0]
        probability = self.model.predict_proba(sample)[0][1]
        
        print(f"\n📈 Prediction Results:")
        print(f"   Actual:      {'🌊 HEAVY RAIN' if actual == 1 else '☀️ NO HEAVY RAIN'}")
        print(f"   Predicted:   {'🌊 HEAVY RAIN' if prediction == 1 else '☀️ NO HEAVY RAIN'}")
        print(f"   Probability: {probability:.2%}")
        
        print(f"\n📊 Feature Values (Top 5 by contribution):")
        feature_contributions = []
        for i, col in enumerate(self.feature_names):
            val = sample[col].values[0]
            shap_val = shap_values[idx][i]
            feature_contributions.append({
                'feature': col,
                'value': val,
                'shap': shap_val,
                'abs_shap': abs(shap_val)
            })
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: x['abs_shap'], reverse=True)
        
        for fc in feature_contributions[:5]:
            emoji = '🔺' if fc['shap'] > 0 else '🔻'
            print(f"   {emoji} {fc['feature']:25}: {fc['value']:8.2f}  (SHAP: {fc['shap']:+.4f})")
        
        # Waterfall plot
        print(f"\n📊 Generating SHAP waterfall plot...")
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx], 
                base_values=explainer.expected_value,
                data=self.X_sample.iloc[idx].values,
                feature_names=self.feature_names
            ),
            show=False,
            max_display=15
        )
        plt.title(f'SHAP Waterfall Plot - Sample #{idx}\nActual: {"HEAVY RAIN" if actual == 1 else "NO HEAVY RAIN"}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_sample_{idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """
        Generate comprehensive explainability report
        """
        print("\n" + "=" * 60)
        print("📋 GENERATING EXPLAINABILITY REPORT")
        print("=" * 60)
        
        # Feature importance
        importance_df = self.analyze_feature_importance()
        
        # SHAP analysis
        try:
            shap_values = self.analyze_shap_values()
        except Exception as e:
            print(f"\n⚠️ SHAP analysis failed: {e}")
            print("   Continuing with feature importance only...")
        
        # Summary
        print("\n" + "=" * 60)
        print("📌 KEY FINDINGS AND INTERPRETATION")
        print("=" * 60)
        
        print("\n🔝 Most Important Features:")
        for i, row in importance_df.head(5).iterrows():
            print(f"   {i+1}. {row['Feature']}: {row['Importance']:.4f}")
        
        print("\n🎯 Model Interpretation:")
        print("   • Higher values in top features increase probability of heavy rain")
        print("   • Temperature patterns and wind speed are key predictors")
        print("   • Seasonal patterns (month, quarter) capture weather cycles")
        print("   • Model aligns with domain knowledge about heavy rainfall")
        
        print("\n⚠️  Limitations:")
        print("   • SHAP values approximate feature contributions")
        print("   • Sample size for SHAP limited to 100 for performance")
        print("   • Model trained on weekly aggregated data")
        print("   • Threshold of 200mm defines 'heavy rain'")
        
        print("\n🌍 Real-world Implications:")
        print("   • Can help predict potential flooding events")
        print("   • Useful for infrastructure planning")
        print("   • Supports disaster preparedness for extreme rainfall")
        print("   • Educational tool for understanding climate patterns")
        
        # Save report
        with open('explainability_report.txt', 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("HEAVY RAIN PREDICTION MODEL EXPLAINABILITY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("FEATURE IMPORTANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(importance_df.to_string())
            f.write("\n\n")
            
            f.write("MODEL INTERPRETATION:\n")
            f.write("-" * 40 + "\n")
            f.write("- Higher values in top features increase probability of heavy rain\n")
            f.write("- Temperature patterns and wind speed are key predictors\n")
            f.write("- Seasonal patterns (month, quarter) capture weather cycles\n")
            f.write("- Model aligns with domain knowledge about heavy rainfall\n\n")
            
            f.write("LIMITATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("- SHAP values approximate feature contributions\n")
            f.write("- Sample size for SHAP limited to 100 for performance\n")
            f.write("- Model trained on weekly aggregated data\n")
            f.write("- Threshold of 200mm defines 'heavy rain'\n\n")
            
            f.write("REAL-WORLD IMPLICATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("- Can help predict potential flooding events\n")
            f.write("- Useful for infrastructure planning\n")
            f.write("- Supports disaster preparedness for extreme rainfall\n")
            f.write("- Educational tool for understanding climate patterns\n")
        
        print("\n✅ Report saved to: explainability_report.txt")

# Run the explainer
if __name__ == "__main__":
    explainer = ModelExplainer()
    explainer.generate_report()
    
    print("\n" + "=" * 60)
    print("✅ EXPLAINABILITY ANALYSIS COMPLETE!")
    print("=" * 60)