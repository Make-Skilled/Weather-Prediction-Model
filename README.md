# Weather Prediction Model

This project implements a machine learning system for weather prediction based on historical data. The system analyzes various weather parameters to make accurate forecasts.

## Features

- Historical weather data generation
- Multiple weather parameters:
  - Temperature (°C)
  - Humidity (%)
  - Wind Speed (km/h)
  - Precipitation (mm)
  - Atmospheric Pressure (hPa)
- Seasonal patterns and realistic weather variations
- One year of sample data

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Generate the sample dataset:
```bash
python generate_weather_data.py
```

## Dataset Structure

The generated dataset (`weather_data.csv`) contains the following columns:
- date: Date of the weather record
- temperature: Temperature in Celsius
- humidity: Humidity percentage
- wind_speed: Wind speed in km/h
- precipitation: Precipitation in mm
- pressure: Atmospheric pressure in hPa

## Data Generation

The sample data is generated with realistic patterns:
- Temperature follows seasonal variations
- Humidity varies around typical values
- Wind speed follows a gamma distribution
- Precipitation includes dry days and rainfall events
- Pressure varies around standard atmospheric pressure

## Next Steps

1. Implement data preprocessing
2. Create feature engineering pipeline
3. Develop machine learning models
4. Implement prediction system
5. Add visualization components