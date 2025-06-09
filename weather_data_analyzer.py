import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from pathlib import Path

class WeatherDataAnalyzer:
    def __init__(self):
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        self.current_data_file = self.data_dir / 'current_weather.csv'
        self.historical_data_file = self.data_dir / 'historical_weather.csv'
        self.analysis_file = self.data_dir / 'weather_analysis.json'
        
        # Initialize data files if they don't exist
        self._initialize_data_files()
    
    def _initialize_data_files(self):
        """Initialize data files with headers if they don't exist"""
        if not self.current_data_file.exists():
            pd.DataFrame(columns=[
                'timestamp', 'city', 'temperature', 'feels_like', 'humidity',
                'pressure', 'wind_speed', 'description', 'icon'
            ]).to_csv(self.current_data_file, index=False)
        
        if not self.historical_data_file.exists():
            pd.DataFrame(columns=[
                'date', 'city', 'temperature', 'humidity', 'wind_speed',
                'pressure', 'description'
            ]).to_csv(self.historical_data_file, index=False)
    
    def store_current_weather(self, weather_data):
        """Store current weather data"""
        try:
            # Create a new row of data
            new_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'city': weather_data['city'],
                'temperature': weather_data['temperature'],
                'feels_like': weather_data['feels_like'],
                'humidity': weather_data['humidity'],
                'pressure': weather_data['pressure'],
                'wind_speed': weather_data['wind_speed'],
                'description': weather_data['description'],
                'icon': weather_data['icon']
            }
            
            # Append to current weather file
            pd.DataFrame([new_data]).to_csv(self.current_data_file, mode='a', header=False, index=False)
            
            # Also store in historical data
            historical_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'city': weather_data['city'],
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'wind_speed': weather_data['wind_speed'],
                'pressure': weather_data['pressure'],
                'description': weather_data['description']
            }
            pd.DataFrame([historical_data]).to_csv(self.historical_data_file, mode='a', header=False, index=False)
            
            return True
        except Exception as e:
            print(f"Error storing weather data: {str(e)}")
            return False
    
    def get_weather_analysis(self, city=None, days=30):
        """Generate weather analysis for the specified city and time period"""
        try:
            # Read historical data
            df = pd.read_csv(self.historical_data_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by city if specified
            if city:
                df = df[df['city'] == city]
            
            # Filter by date range
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=days)
            df = df[df['date'] >= start_date]
            
            if len(df) == 0:
                return {
                    'status': 'error',
                    'message': 'No data available for the specified period'
                }
            
            # Calculate statistics
            analysis = {
                'temperature': {
                    'mean': round(df['temperature'].mean(), 1),
                    'min': round(df['temperature'].min(), 1),
                    'max': round(df['temperature'].max(), 1),
                    'std': round(df['temperature'].std(), 1)
                },
                'humidity': {
                    'mean': round(df['humidity'].mean(), 1),
                    'min': round(df['humidity'].min(), 1),
                    'max': round(df['humidity'].max(), 1),
                    'std': round(df['humidity'].std(), 1)
                },
                'wind_speed': {
                    'mean': round(df['wind_speed'].mean(), 1),
                    'min': round(df['wind_speed'].min(), 1),
                    'max': round(df['wind_speed'].max(), 1),
                    'std': round(df['wind_speed'].std(), 1)
                },
                'pressure': {
                    'mean': round(df['pressure'].mean(), 1),
                    'min': round(df['pressure'].min(), 1),
                    'max': round(df['pressure'].max(), 1),
                    'std': round(df['pressure'].std(), 1)
                },
                'weather_conditions': df['description'].value_counts().to_dict(),
                'data_points': len(df),
                'date_range': {
                    'start': df['date'].min().strftime('%Y-%m-%d'),
                    'end': df['date'].max().strftime('%Y-%m-%d')
                }
            }
            
            # Save analysis to file
            with open(self.analysis_file, 'w') as f:
                json.dump(analysis, f, indent=4)
            
            return {
                'status': 'success',
                'data': analysis
            }
            
        except Exception as e:
            print(f"Error generating weather analysis: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_recent_weather(self, city=None, limit=10):
        """Get recent weather data"""
        try:
            df = pd.read_csv(self.current_data_file)
            if city:
                df = df[df['city'] == city]
            return df.tail(limit).to_dict('records')
        except Exception as e:
            print(f"Error getting recent weather: {str(e)}")
            return []
    
    def get_historical_data(self, city=None, start_date=None, end_date=None):
        """Get historical weather data with optional filters"""
        try:
            df = pd.read_csv(self.historical_data_file)
            df['date'] = pd.to_datetime(df['date'])
            
            if city:
                df = df[df['city'] == city]
            if start_date:
                df = df[df['date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['date'] <= pd.to_datetime(end_date)]
            
            return df.to_dict('records')
        except Exception as e:
            print(f"Error getting historical data: {str(e)}")
            return [] 