"""
Model Training for Weather Prediction using XGBoost
XGBoost is chosen as it's a powerful ensemble method not covered in lectures
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, f1_score, precision_score, recall_score)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class WeatherModelTrainer:
    def __init__(self, data_path='data/weekly_weather_data.csv'):
        """
        Initialize the model trainer
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None
        self.best_params = None
        self.feature_names = None
        
    def load_data(self):
        """
        Load the dataset and create target variable
        """
        print("=" * 60)
        print("STEP 1: Loading Data")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Check for any remaining missing values
        missing = self.df.isnull().sum().sum()
        if missing > 0:
            print(f"\n⚠️  Warning: {missing} missing values found. Dropping rows...")
            self.df = self.df.dropna()
            print(f"New shape: {self.df.shape}")

        # Create target variable for heavy rain - using higher threshold for better balance
        if 'RainTomorrow' not in self.df.columns:
            print("\n📊 Creating target variable 'RainTomorrow' (heavy rain)...")
            threshold = 200.0  # Higher threshold for more balanced classes
            self.df['RainTomorrow'] = np.where(self.df['rain_sum'] > threshold, 1, 0)
            print(f"   Threshold used: {threshold}mm")
        
        # Display class distribution
        print(f"\nClass distribution:")
        class_dist = self.df['RainTomorrow'].value_counts()
        for cls, count in class_dist.items():
            percentage = count/len(self.df)*100
            print(f"  Class {cls}: {count} ({percentage:.1f}%)")
        
        return self.df
    
    def prepare_features(self):
        """
        Separate features and target - handle categorical variables properly
        Remove highly correlated features to prevent overfitting
        """
        print("\n" + "=" * 60)
        print("STEP 2: Preparing Features and Target")
        print("=" * 60)
        
        # Create a copy to avoid modifying original
        df_features = self.df.copy()
        
        # Handle time column - extract useful features
        if 'time' in df_features.columns:
            print("\n🕐 Processing time column...")
            df_features['time'] = pd.to_datetime(df_features['time'])
            df_features['year'] = df_features['time'].dt.year
            df_features['month'] = df_features['time'].dt.month
            df_features['day'] = df_features['time'].dt.day
            df_features['day_of_week'] = df_features['time'].dt.dayofweek
            df_features['week_of_year'] = df_features['time'].dt.isocalendar().week
            df_features['quarter'] = df_features['time'].dt.quarter
            df_features = df_features.drop('time', axis=1)
            print(f"   ✓ Extracted time features: year, month, day, day_of_week, week_of_year, quarter")
        
        # Handle categorical columns - encode them
        categorical_cols = ['country', 'city']
        for col in categorical_cols:
            if col in df_features.columns:
                print(f"\n🔤 Encoding {col}...")
                df_features[col] = pd.Categorical(df_features[col]).codes
                print(f"   ✓ Encoded {col}")
        
        # Create additional useful features - WITHOUT using rain_sum directly
        print("\n🔧 Creating additional features (avoiding direct correlations)...")
        
        # Temperature-based features
        if 'temperature_2m_max' in df_features.columns and 'temperature_2m_min' in df_features.columns:
            df_features['temp_range'] = df_features['temperature_2m_max'] - df_features['temperature_2m_min']
            print("   ✓ Created temperature range feature")
        
        if 'temperature_2m_mean' in df_features.columns:
            df_features['temp_squared'] = df_features['temperature_2m_mean'] ** 2
            df_features['temp_cubed'] = df_features['temperature_2m_mean'] ** 3
            print("   ✓ Created temperature polynomial features")
        
        # Wind-based features
        if 'windspeed_10m_max' in df_features.columns:
            df_features['wind_squared'] = df_features['windspeed_10m_max'] ** 2
            df_features['wind_log'] = np.log1p(df_features['windspeed_10m_max'])
            print("   ✓ Created wind transformation features")
        
        # Weather code features (categorical already encoded)
        if 'weathercode' in df_features.columns:
            # Create weather code categories (based on WMO codes)
            # 0-19: No precipitation, 20-29: Showers, 30-39: Dust, 40-49: Fog, 50-69: Rain, 70-79: Snow, 80-99: Thunderstorms
            df_features['weather_category'] = pd.cut(
                df_features['weathercode'], 
                bins=[-1, 19, 29, 39, 49, 69, 79, 99], 
                labels=[0, 1, 2, 3, 4, 5, 6]
            ).astype(int)
            print("   ✓ Created weather category feature")
        
        # Interaction features
        if 'temperature_2m_mean' in df_features.columns and 'windspeed_10m_max' in df_features.columns:
            df_features['temp_wind_interaction'] = df_features['temperature_2m_mean'] * df_features['windspeed_10m_max']
            print("   ✓ Created temperature-wind interaction")
        
        # Rolling averages (simulated with current values as proxy)
        if 'temperature_2m_mean' in df_features.columns:
            df_features['temp_rolling_avg'] = df_features['temperature_2m_mean'].rolling(window=3, min_periods=1).mean()
            print("   ✓ Created temperature rolling average")
        
        # Fill any NaN values from rolling operations
        df_features = df_features.fillna(0)
        
        # Drop the original highly correlated features that cause overfitting
        features_to_drop = ['rain_sum', 'precipitation_sum']  # Remove these direct predictors
        for col in features_to_drop:
            if col in df_features.columns:
                df_features = df_features.drop(col, axis=1)
                print(f"   ✓ Dropped {col} to prevent overfitting")
        
        # Also drop any other potentially leaky features
        if 'heavy_rain' in df_features.columns:
            df_features = df_features.drop('heavy_rain', axis=1)
            print(f"   ✓ Dropped heavy_rain indicator")
        
        if 'precip_ratio' in df_features.columns:
            df_features = df_features.drop('precip_ratio', axis=1)
            print(f"   ✓ Dropped precip_ratio")
        
        # Separate features and target
        X = df_features.drop('RainTomorrow', axis=1)
        y = df_features['RainTomorrow']
        
        self.feature_names = list(X.columns)
        
        print(f"\n✅ Final feature set:")
        print(f"   Features shape: {X.shape}")
        print(f"   Number of features: {len(self.feature_names)}")
        print(f"   Target shape: {y.shape}")
        print(f"   Features: {self.feature_names}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.25, random_state=42):
        """
        Split data into train, validation, and test sets with stratification
        """
        print("\n" + "=" * 60)
        print("STEP 3: Train/Validation/Test Split")
        print("=" * 60)
        
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y, shuffle=True
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, 
            random_state=random_state, stratify=y_train_val
        )
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        print(f"\nTraining set class distribution:")
        train_dist = pd.Series(y_train).value_counts()
        for cls, count in train_dist.items():
            print(f"  Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_xgboost(self):
        """
        Train XGBoost model with strong regularization to prevent overfitting
        """
        print("\n" + "=" * 60)
        print("STEP 4: Training XGBoost Model (with Strong Regularization)")
        print("=" * 60)
        
        print("\n🔍 Why XGBoost?")
        print("   ✓ Gradient boosting algorithm that builds trees sequentially")
        print("   ✓ Each new tree corrects errors of previous trees")
        print("   ✓ Uses regularization to prevent overfitting")
        print("   ✓ Handles missing values automatically")
        print("   ✓ Provides feature importance scores")
        print("   ✓ Handles non-linear relationships well")
        
        # Combine train and validation
        X_train_full = pd.concat([self.X_train, self.X_val])
        y_train_full = pd.concat([self.y_train, self.y_val])
        
        # Calculate scale_pos_weight for class imbalance
        neg_count = (y_train_full == 0).sum()
        pos_count = (y_train_full == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        print(f"\n⚖️  Class imbalance ratio: {scale_pos_weight:.2f}")
        print(f"   Using scale_pos_weight = {scale_pos_weight:.2f}")
        
        # Train model with strong regularization
        self.model = xgb.XGBClassifier(
            # Reduce model complexity
            n_estimators=100,
            max_depth=2,                    # Reduced depth
            learning_rate=0.05,              # Lower learning rate
            
            # Strong regularization
            reg_alpha=2.0,                   # L1 regularization
            reg_lambda=3.0,                   # L2 regularization
            gamma=0.1,                        # Minimum loss reduction
            
            # Add randomness to prevent overfitting
            subsample=0.6,                    # Use only 60% of data
            colsample_bytree=0.6,              # Use only 60% of features
            colsample_bylevel=0.6,             # Subsample at each level
            
            # Conservative splitting
            min_child_weight=5,                 # Minimum child weight
            
            # Other parameters
            objective='binary:logistic',
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Train the model with early stopping
        self.model.fit(
            X_train_full, 
            y_train_full,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )
        
        self.best_params = {
            'n_estimators': 100,
            'max_depth': 2,
            'learning_rate': 0.05,
            'reg_alpha': 2.0,
            'reg_lambda': 3.0,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'min_child_weight': 5,
            'gamma': 0.1
        }
        
        print("\n✅ Model trained with strong regularization!")
        print(f"   Final number of trees: {self.model.n_estimators_ if hasattr(self.model, 'n_estimators_') else 100}")
        
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate the model on validation and test sets
        """
        print("\n" + "=" * 60)
        print("STEP 5: Model Evaluation")
        print("=" * 60)
        
        # Test set predictions
        y_test_pred = self.model.predict(self.X_test)
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_f1 = f1_score(self.y_test, y_test_pred, zero_division=0)
        test_auc = roc_auc_score(self.y_test, y_test_proba)
        test_precision = precision_score(self.y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(self.y_test, y_test_pred, zero_division=0)
        
        print("\n📊 Test Set Performance:")
        print(f"   Accuracy:  {test_accuracy:.4f}")
        print(f"   F1 Score:  {test_f1:.4f}")
        print(f"   ROC-AUC:   {test_auc:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall:    {test_recall:.4f}")
        
        # Classification report
        print("\n📋 Test Set Classification Report:")
        print(classification_report(self.y_test, y_test_pred, 
                                   target_names=['No Heavy Rain', 'Heavy Rain'],
                                   zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Heavy Rain', 'Heavy Rain'],
                    yticklabels=['No Heavy Rain', 'Heavy Rain'])
        plt.title(f'Confusion Matrix - Test Set\n(Accuracy: {test_accuracy:.3f}, F1: {test_f1:.3f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()  # Don't show, just save
        
        return {
            'test': {
                'accuracy': test_accuracy,
                'f1': test_f1,
                'roc_auc': test_auc,
                'precision': test_precision,
                'recall': test_recall
            }
        }
    
    def plot_feature_importance(self):
        """
        Plot and analyze feature importance
        """
        print("\n" + "=" * 60)
        print("STEP 6: Feature Importance Analysis")
        print("=" * 60)
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\n🔝 Top 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, min(10, len(importance_df))))
        bars = plt.barh(importance_df['Feature'][:10][::-1], 
                       importance_df['Importance'][:10][::-1], 
                       color=colors[::-1])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, importance_df['Importance'][:10][::-1])):
            plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save importance to CSV
        importance_df.to_csv('feature_importance.csv', index=False)
        print("\n✅ Feature importance saved to 'feature_importance.csv'")
        
        return importance_df
    
    def save_model(self):
        """
        Save the trained model and metadata
        """
        print("\n" + "=" * 60)
        print("STEP 7: Saving Model")
        print("=" * 60)
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'XGBoost Classifier',
            'threshold': 200.0
        }
        
        joblib.dump(model_data, 'xgboost_model.pkl')
        print("✅ Model saved to: xgboost_model.pkl")
        print(f"   Model type: XGBoost Classifier")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Best params: {self.best_params}")
    
    def run_pipeline(self):
        """
        Run the complete training pipeline
        """
        print("\n" + "=" * 60)
        print("WEATHER PREDICTION MODEL TRAINING")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        self.load_data()
        X, y = self.prepare_features()
        self.split_data(X, y)
        self.train_xgboost()
        metrics = self.evaluate_model()
        self.plot_feature_importance()
        self.save_model()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        return metrics

# Run the pipeline if script is executed directly
if __name__ == "__main__":
    trainer = WeatherModelTrainer()
    metrics = trainer.run_pipeline()
    
    print("\n✅ Model training completed successfully!")
    print(f"   Test Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"   Test F1 Score: {metrics['test']['f1']:.4f}")
    print(f"   Test ROC-AUC: {metrics['test']['roc_auc']:.4f}")