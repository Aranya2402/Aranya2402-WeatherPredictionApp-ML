"""
Flask Frontend for Weather Prediction
"""

from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Backend URL
BACKEND_URL = 'http://localhost:8000'

@app.route('/')
def index():
    """Render the main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    """Get prediction from backend"""
    try:
        # Get form data
        data = {
            "weathercode": float(request.form['weathercode']),
            "temperature_2m_max": float(request.form['temperature_2m_max']),
            "temperature_2m_min": float(request.form['temperature_2m_min']),
                        "temperature_2m_mean": float(request.form['temperature_2m_mean']),  # ADD THIS

            "precipitation_sum": float(request.form['precipitation_sum']),
            "rain_sum": float(request.form['rain_sum']),
            "windspeed_10m_max": float(request.form['windspeed_10m_max']),
            "country": request.form['country'],
            "city": request.form['city']
        }
        
        # Send to backend
        response = requests.post(f"{BACKEND_URL}/predict", json=data, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            return render_template('index.html', result=result, input_data=data)
        else:
            return render_template('index.html', error=f"Backend error: {response.status_code}", input_data=data)
            
    except requests.exceptions.ConnectionError:
        return render_template('index.html', error="Cannot connect to backend. Make sure it's running on port 8000")
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Frontend Server")
    print("=" * 50)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Frontend URL: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)