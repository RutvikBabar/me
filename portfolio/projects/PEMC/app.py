from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import numpy as np
from pemc_core import PEMCBlackScholes
import threading
import time

app = Flask(__name__)
CORS(app)

# Global PEMC model instance
pemc_model = None
model_training_status = {"status": "not_started", "progress": 0}

def train_model_background():
    """Train the PEMC model in background"""
    global pemc_model, model_training_status
    
    try:
        model_training_status["status"] = "training"
        model_training_status["progress"] = 10
        
        pemc_model = PEMCBlackScholes()
        model_training_status["progress"] = 30
        
        # Train the model
        pemc_model.train_prediction_model(n_training_samples=30000)
        
        model_training_status["status"] = "completed"
        model_training_status["progress"] = 100
        print("PEMC model training completed!")
        
    except Exception as e:
        model_training_status["status"] = "error"
        model_training_status["error"] = str(e)
        print(f"Error training model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/model_status')
def model_status():
    """Get current model training status"""
    return jsonify(model_training_status)

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Start model training"""
    global model_training_status
    
    if model_training_status["status"] == "training":
        return jsonify({"error": "Model is already training"}), 400
    
    # Start training in background thread
    training_thread = threading.Thread(target=train_model_background)
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({"message": "Model training started"})

@app.route('/api/calculate', methods=['POST'])
def calculate():
    """Calculate option prices using PEMC"""
    global pemc_model
    
    if pemc_model is None or not pemc_model.is_trained:
        return jsonify({"error": "Model not trained yet"}), 400
    
    try:
        data = request.json
        
        S = float(data['S'])
        K = float(data['K'])
        T = float(data['T'])
        r = float(data['r']) / 100  # Convert percentage to decimal
        sigma = float(data['sigma']) / 100  # Convert percentage to decimal
        n_expensive = int(data.get('n_expensive', 1000))
        n_cheap = int(data.get('n_cheap', 10000))
        
        # Calculate traditional Black-Scholes
        traditional_bs = pemc_model.black_scholes_price(S, K, T, r, sigma)
        
        # Calculate PEMC estimate
        pemc_result = pemc_model.pemc_estimate(S, K, T, r, sigma, n_expensive, n_cheap)
        
        # Calculate intrinsic values
        call_intrinsic = max(S - K, 0)
        put_intrinsic = max(K - S, 0)
        
        response = {
            'success': True,
            'traditional': {
                'call_price': traditional_bs['call_price'],
                'put_price': traditional_bs['put_price'],
                'call_intrinsic': call_intrinsic,
                'put_intrinsic': put_intrinsic,
                'call_time_value': traditional_bs['call_price'] - call_intrinsic,
                'put_time_value': traditional_bs['put_price'] - put_intrinsic,
                'greeks': {
                    'call_delta': traditional_bs['call_delta'],
                    'put_delta': traditional_bs['put_delta'],
                    'gamma': traditional_bs['gamma'],
                    'vega': traditional_bs['vega'],
                    'call_theta': traditional_bs['call_theta'],
                    'put_theta': traditional_bs['put_theta'],
                    'call_rho': traditional_bs['call_rho'],
                    'put_rho': traditional_bs['put_rho']
                }
            },
            'pemc': {
                'call_price': pemc_result.pemc_estimate,
                'put_price': pemc_result.pemc_estimate,  # For Asian options, we're pricing calls
                'traditional_mc': pemc_result.traditional_mc,
                'variance_reduction': pemc_result.variance_reduction,
                'confidence_interval': pemc_result.confidence_interval,
                'correlation': pemc_result.correlation,
                'cost_ratio': pemc_result.cost_ratio,
                'ml_confidence': pemc_result.ml_confidence
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/sensitivity', methods=['POST'])
def sensitivity_analysis():
    """Perform sensitivity analysis"""
    global pemc_model
    
    if pemc_model is None or not pemc_model.is_trained:
        return jsonify({"error": "Model not trained yet"}), 400
    
    try:
        data = request.json
        
        S = float(data['S'])
        K = float(data['K'])
        T = float(data['T'])
        r = float(data['r']) / 100
        sigma = float(data['sigma']) / 100
        parameter = data.get('parameter', 'S')
        
        results = pemc_model.sensitivity_analysis(S, K, T, r, sigma, parameter, points=15)
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting PEMC Options Pricing Server...")
    print("Training will begin automatically...")
    
    # Start model training immediately
    training_thread = threading.Thread(target=train_model_background)
    training_thread.daemon = True
    training_thread.start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
