from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from weather_prediction_model import WeatherPredictionModel
from weather_data_analyzer import WeatherDataAnalyzer
import pandas as pd
import json
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)
CORS(app)

# Initialize the weather prediction model and data analyzer
model = None
analyzer = WeatherDataAnalyzer()

# OpenWeather API configuration
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
DEFAULT_CITY = os.getenv('DEFAULT_CITY')

def get_weather_data(city=DEFAULT_CITY):
    """Get real-time weather data from OpenWeather API"""
    try:
        if not OPENWEATHER_API_KEY:
            print("Error: OpenWeather API key is not set")
            return {
                'error': 'API_KEY_MISSING',
                'message': 'OpenWeather API key is not set. Please add your API key to the .env file.'
            }

        # Current weather
        current_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        current_response = requests.get(current_url)
        
        if current_response.status_code == 401:
            print("Error: Invalid API key")
            return {
                'error': 'API_KEY_INVALID',
                'message': 'Invalid OpenWeather API key. Please check your API key in the .env file.'
            }
        elif current_response.status_code != 200:
            print(f"Error fetching current weather: {current_response.status_code}")
            print(f"Response: {current_response.text}")
            return {
                'error': 'API_ERROR',
                'message': f'Error fetching weather data: {current_response.text}'
            }
            
        current_data = current_response.json()

        # 5-day forecast
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        forecast_response = requests.get(forecast_url)
        
        if forecast_response.status_code == 401:
            print("Error: Invalid API key")
            return {
                'error': 'API_KEY_INVALID',
                'message': 'Invalid OpenWeather API key. Please check your API key in the .env file.'
            }
        elif forecast_response.status_code != 200:
            print(f"Error fetching forecast: {forecast_response.status_code}")
            print(f"Response: {forecast_response.text}")
            return {
                'error': 'API_ERROR',
                'message': f'Error fetching forecast data: {forecast_response.text}'
            }
            
        forecast_data = forecast_response.json()

        return {
            'current': current_data,
            'forecast': forecast_data
        }
    except requests.exceptions.RequestException as e:
        print(f"Network error: {str(e)}")
        return {
            'error': 'NETWORK_ERROR',
            'message': f'Network error: {str(e)}'
        }
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {
            'error': 'UNKNOWN_ERROR',
            'message': f'An unexpected error occurred: {str(e)}'
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analysis')
def analysis_page():
    return render_template('analysis.html')

@app.route('/api/initialize', methods=['GET'])
def initialize_model():
    global model
    try:
        # Load and prepare the dataset
        df = pd.read_csv('weather_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Create and train the model
        model = WeatherPredictionModel()
        model.train(df)
        
        return jsonify({
            'status': 'success',
            'message': 'Model initialized and trained successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict', methods=['GET'])
def predict():
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not initialized. Please call /api/initialize first'
        }), 400
    
    try:
        # Load the latest data
        df = pd.read_csv('weather_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Get predictions
        predictions = model.predict_next_day(df)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/weather/current', methods=['GET'])
def get_current_weather():
    city = request.args.get('city', DEFAULT_CITY)
    weather_data = get_weather_data(city)
    
    if weather_data is None:
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch weather data. Please check if the API key is set correctly.'
        }), 500
    
    try:
        current = weather_data['current']
        data = {
            'temperature': round(current['main']['temp'], 1),
            'feels_like': round(current['main']['feels_like'], 1),
            'humidity': current['main']['humidity'],
            'pressure': current['main']['pressure'],
            'wind_speed': round(current['wind']['speed'] * 3.6, 1),  # Convert m/s to km/h
            'description': current['weather'][0]['description'],
            'icon': current['weather'][0]['icon'],
            'city': current['name'],
            'country': current['sys']['country'],
            'timestamp': datetime.fromtimestamp(current['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Store the weather data
        analyzer.store_current_weather(data)
        
        return jsonify({
            'status': 'success',
            'data': data
        })
    except KeyError as e:
        print(f"Error parsing weather data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Error parsing weather data. Please check the API response format.'
        }), 500
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred while processing weather data.'
        }), 500

@app.route('/api/weather/forecast', methods=['GET'])
def get_weather_forecast():
    city = request.args.get('city', DEFAULT_CITY)
    weather_data = get_weather_data(city)
    
    if weather_data:
        forecast = weather_data['forecast']
        daily_forecasts = []
        
        # Group forecast by day
        current_date = None
        daily_data = {}
        
        for item in forecast['list']:
            date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
            
            if date != current_date:
                if current_date and daily_data:
                    daily_forecasts.append(daily_data)
                current_date = date
                daily_data = {
                    'date': date,
                    'temperature': round(item['main']['temp'], 1),
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': round(item['wind']['speed'] * 3.6, 1),
                    'description': item['weather'][0]['description'],
                    'icon': item['weather'][0]['icon']
                }
            else:
                # Update with latest data for the day
                daily_data.update({
                    'temperature': round(item['main']['temp'], 1),
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': round(item['wind']['speed'] * 3.6, 1),
                    'description': item['weather'][0]['description'],
                    'icon': item['weather'][0]['icon']
                })
        
        if daily_data:
            daily_forecasts.append(daily_data)
        
        return jsonify({
            'status': 'success',
            'data': daily_forecasts
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch forecast data'
        }), 500

@app.route('/api/weather/analysis', methods=['GET'])
def get_weather_analysis():
    city = request.args.get('city')
    days = int(request.args.get('days', 30))
    
    analysis = analyzer.get_weather_analysis(city, days)
    return jsonify(analysis)

@app.route('/api/weather/historical', methods=['GET'])
def get_historical_weather():
    city = request.args.get('city')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    data = analyzer.get_historical_data(city, start_date, end_date)
    return jsonify({
        'status': 'success',
        'data': data
    })

@app.route('/api/weather/recent', methods=['GET'])
def get_recent_weather():
    city = request.args.get('city')
    limit = int(request.args.get('limit', 10))
    
    data = analyzer.get_recent_weather(city, limit)
    return jsonify({
        'status': 'success',
        'data': data
    })

@app.route('/api/weather/alerts', methods=['GET'])
def get_weather_alerts():
    city = request.args.get('city', DEFAULT_CITY)
    weather_data = get_weather_data(city)
    
    if weather_data:
        current = weather_data['current']
        alerts = []
        
        # Temperature alerts
        temp = current['main']['temp']
        if temp > 30:
            alerts.append({
                'type': 'warning',
                'message': 'High temperature alert: Stay hydrated and avoid prolonged sun exposure.'
            })
        elif temp < 10:
            alerts.append({
                'type': 'info',
                'message': 'Low temperature alert: Dress warmly and be cautious of frost.'
            })
        
        # Humidity alerts
        humidity = current['main']['humidity']
        if humidity > 80:
            alerts.append({
                'type': 'warning',
                'message': 'High humidity alert: Increased risk of heat-related illnesses.'
            })
        
        # Wind alerts
        wind_speed = current['wind']['speed'] * 3.6  # Convert m/s to km/h
        if wind_speed > 20:
            alerts.append({
                'type': 'warning',
                'message': 'Strong wind alert: Secure outdoor objects and be cautious.'
            })
        
        if not alerts:
            alerts.append({
                'type': 'success',
                'message': 'No weather alerts at this time.'
            })
        
        return jsonify({
            'status': 'success',
            'data': alerts
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to fetch weather alerts'
        }), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, port=5000) 