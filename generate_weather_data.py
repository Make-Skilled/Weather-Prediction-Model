import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_weather_data(num_days=365):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(num_days)]
    
    # Generate temperature data with seasonal patterns
    base_temp = 15  # Base temperature in Celsius
    seasonal_effect = 10 * np.sin(np.linspace(0, 4*np.pi, num_days))  # Seasonal variation
    daily_variation = np.random.normal(0, 2, num_days)  # Daily random variation
    temperatures = base_temp + seasonal_effect + daily_variation
    
    # Generate humidity data (0-100%)
    humidity = np.random.normal(60, 15, num_days)
    humidity = np.clip(humidity, 0, 100)
    
    # Generate wind speed (km/h)
    wind_speed = np.random.gamma(2, 2, num_days)
    
    # Generate precipitation (mm)
    precipitation = np.random.exponential(2, num_days)
    # Make some days have no precipitation
    precipitation = np.where(np.random.random(num_days) < 0.7, 0, precipitation)
    
    # Generate pressure (hPa)
    pressure = np.random.normal(1013, 5, num_days)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'temperature': np.round(temperatures, 1),
        'humidity': np.round(humidity, 1),
        'wind_speed': np.round(wind_speed, 1),
        'precipitation': np.round(precipitation, 1),
        'pressure': np.round(pressure, 1)
    })
    
    return df

if __name__ == "__main__":
    # Generate one year of weather data
    weather_data = generate_weather_data()
    
    # Save to CSV
    weather_data.to_csv('weather_data.csv', index=False)
    print("Sample weather dataset has been generated and saved to 'weather_data.csv'") 