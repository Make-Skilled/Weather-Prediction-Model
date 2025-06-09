import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class WeatherPredictionModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'pressure']
        
    def prepare_data(self, df, target_column, n_steps=7):
        """Prepare data for time series prediction"""
        # Create lag features
        for i in range(1, n_steps + 1):
            for col in self.feature_columns:
                df[f'{col}_lag_{i}'] = df[col].shift(i)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col.endswith(('lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7'))]
        X = df[feature_cols]
        y = df[target_column]
        
        return X, y
    
    def train(self, df):
        """Train models for each weather parameter"""
        for target in self.feature_columns:
            print(f"\nTraining model for {target}...")
            
            # Prepare data
            X, y = self.prepare_data(df, target)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"R2 Score: {r2:.2f}")
            
            # Store model and scaler
            self.models[target] = model
            self.scalers[target] = scaler
            
            # Plot predictions vs actual
            self.plot_predictions(y_test, y_pred, target)
    
    def plot_predictions(self, y_true, y_pred, target):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted {target.capitalize()}')
        plt.tight_layout()
        plt.savefig(f'{target}_predictions.png')
        plt.close()
    
    def predict_next_day(self, df):
        """Predict weather conditions for the next day"""
        predictions = {}
        
        for target in self.feature_columns:
            # Prepare data for the last 7 days
            X, _ = self.prepare_data(df, target)
            if len(X) > 0:
                # Get the last row of features
                last_features = X.iloc[-1:].values
                
                # Scale features
                last_features_scaled = self.scalers[target].transform(last_features)
                
                # Make prediction
                pred = self.models[target].predict(last_features_scaled)[0]
                predictions[target] = round(pred, 2)
        
        return predictions

def main():
    # Load the dataset
    df = pd.read_csv('weather_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Create and train the model
    model = WeatherPredictionModel()
    model.train(df)
    
    # Make predictions for the next day
    next_day_predictions = model.predict_next_day(df)
    
    print("\nPredictions for the next day:")
    for param, value in next_day_predictions.items():
        print(f"{param.capitalize()}: {value}")

if __name__ == "__main__":
    main() 