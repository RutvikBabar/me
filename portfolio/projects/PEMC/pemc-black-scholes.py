import numpy as np
import math
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

@dataclass
class MarketFeatures:
    """Market features for PEMC prediction"""
    brownian_increments: np.ndarray  # Sum of Brownian increments (X)
    spot_price: float
    strike_price: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float

class PEMCBlackScholes:
    """PEMC-Enhanced Black-Scholes Implementation"""
    
    def __init__(self):
        self.prediction_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function for standard normal distribution"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Probability density function for standard normal distribution"""
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)
    
    def black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
        """Traditional Black-Scholes pricing"""
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        call_price = S * self._normal_cdf(d1) - K * math.exp(-r * T) * self._normal_cdf(d2)
        put_price = K * math.exp(-r * T) * self._normal_cdf(-d2) - S * self._normal_cdf(-d1)
        
        # Greeks
        call_delta = self._normal_cdf(d1)
        put_delta = call_delta - 1
        gamma = self._normal_pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * self._normal_pdf(d1) * math.sqrt(T) / 100
        call_theta = (-S * self._normal_pdf(d1) * sigma / (2 * math.sqrt(T)) - 
                     r * K * math.exp(-r * T) * self._normal_cdf(d2)) / 365
        put_theta = (-S * self._normal_pdf(d1) * sigma / (2 * math.sqrt(T)) + 
                    r * K * math.exp(-r * T) * self._normal_cdf(-d2)) / 365
        call_rho = K * T * math.exp(-r * T) * self._normal_cdf(d2) / 100
        put_rho = -K * T * math.exp(-r * T) * self._normal_cdf(-d2) / 100
        
        return {
            'call_price': call_price,
            'put_price': put_price,
            'call_delta': call_delta,
            'put_delta': put_delta,
            'gamma': gamma,
            'vega': vega,
            'call_theta': call_theta,
            'put_theta': put_theta,
            'call_rho': call_rho,
            'put_rho': put_rho
        }
    
    def generate_training_data(self, n_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for PEMC predictor"""
        print(f"Generating {n_samples} training samples...")
        
        features = []
        labels = []
        
        for i in range(n_samples):
            if i % 10000 == 0:
                print(f"Generated {i}/{n_samples} samples")
            
            # Sample parameters from realistic ranges
            S = np.random.uniform(80, 120)
            K = np.random.uniform(90, 110)
            T = np.random.uniform(0.1, 1.0)
            r = np.random.uniform(0.01, 0.08)
            sigma = np.random.uniform(0.1, 0.4)
            
            # Generate correlated Brownian motion path
            dt = T / 252  # Daily steps
            n_steps = int(T * 252)
            
            # Simulate stock price path using GBM
            brownian_increments = np.random.normal(0, math.sqrt(dt), n_steps)
            cumulative_brownian = np.sum(brownian_increments)
            
            # Calculate option payoff (call option)
            St_final = S * math.exp((r - 0.5 * sigma**2) * T + sigma * cumulative_brownian)
            call_payoff = max(St_final - K, 0) * math.exp(-r * T)
            
            # Features: [S, K, T, r, sigma, cumulative_brownian]
            feature_vector = [S, K, T, r, sigma, cumulative_brownian]
            features.append(feature_vector)
            labels.append(call_payoff)
        
        return np.array(features), np.array(labels)
    
    def train_prediction_model(self, n_training_samples: int = 100000):
        """Train the ML prediction model for PEMC"""
        print("Training PEMC prediction model...")
        
        # Generate training data
        X_train, y_train = self.generate_training_data(n_training_samples)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest model (as suggested in the paper for robustness)
        self.prediction_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        self.prediction_model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate training accuracy
        y_pred = self.prediction_model.predict(X_train_scaled)
        mse = np.mean((y_train - y_pred)**2)
        print(f"Training completed. MSE: {mse:.6f}")
        
        return mse
    
    def pemc_estimate(self, S: float, K: float, T: float, r: float, sigma: float, 
                     n_expensive: int = 1000, n_cheap: int = 10000) -> Dict:
        """PEMC estimation following the paper's methodology"""
        
        if not self.is_trained:
            print("Model not trained. Training with default parameters...")
            self.train_prediction_model()
        
        # Step 1: Generate n expensive coupled samples (Y_i, X_i)
        expensive_samples = []
        expensive_features = []
        
        dt = T / 252
        n_steps = int(T * 252)
        
        for i in range(n_expensive):
            # Generate Brownian path
            brownian_increments = np.random.normal(0, math.sqrt(dt), n_steps)
            cumulative_brownian = np.sum(brownian_increments)
            
            # Calculate true option payoff (expensive evaluation)
            St_final = S * math.exp((r - 0.5 * sigma**2) * T + sigma * cumulative_brownian)
            true_payoff = max(St_final - K, 0) * math.exp(-r * T)
            
            # ML prediction (cheap evaluation)
            feature_vector = np.array([[S, K, T, r, sigma, cumulative_brownian]])
            feature_scaled = self.scaler.transform(feature_vector)
            ml_prediction = self.prediction_model.predict(feature_scaled)[0]
            
            expensive_samples.append(true_payoff - ml_prediction)
            expensive_features.append([S, K, T, r, sigma, cumulative_brownian])
        
        # Step 2: Generate N cheap independent samples (X̃_j)
        cheap_predictions = []
        
        for j in range(n_cheap):
            # Generate independent Brownian increment
            brownian_increments = np.random.normal(0, math.sqrt(dt), n_steps)
            cumulative_brownian = np.sum(brownian_increments)
            
            # ML prediction only (very cheap)
            feature_vector = np.array([[S, K, T, r, sigma, cumulative_brownian]])
            feature_scaled = self.scaler.transform(feature_vector)
            ml_prediction = self.prediction_model.predict(feature_scaled)[0]
            
            cheap_predictions.append(ml_prediction)
        
        # Step 3: PEMC estimator
        expensive_term = np.mean(expensive_samples)
        cheap_term = np.mean(cheap_predictions)
        pemc_price = expensive_term + cheap_term
        
        # Calculate traditional Black-Scholes for comparison
        traditional_bs = self.black_scholes_price(S, K, T, r, sigma)
        
        # Calculate standard MC for comparison
        mc_samples = []
        for i in range(n_expensive):
            brownian_increments = np.random.normal(0, math.sqrt(dt), n_steps)
            cumulative_brownian = np.sum(brownian_increments)
            St_final = S * math.exp((r - 0.5 * sigma**2) * T + sigma * cumulative_brownian)
            mc_payoff = max(St_final - K, 0) * math.exp(-r * T)
            mc_samples.append(mc_payoff)
        
        mc_price = np.mean(mc_samples)
        mc_std = np.std(mc_samples) / math.sqrt(n_expensive)
        
        # Calculate PEMC variance components
        expensive_var = np.var(expensive_samples) / n_expensive
        cheap_var = np.var(cheap_predictions) / n_cheap
        pemc_var = expensive_var + cheap_var
        pemc_std = math.sqrt(pemc_var)
        
        return {
            'pemc_price': pemc_price,
            'pemc_std': pemc_std,
            'traditional_bs': traditional_bs['call_price'],
            'monte_carlo': mc_price,
            'mc_std': mc_std,
            'variance_reduction': (mc_std**2 - pemc_std**2) / mc_std**2 * 100,
            'n_expensive': n_expensive,
            'n_cheap': n_cheap,
            'greeks': traditional_bs
        }
    
    def sensitivity_analysis(self, S: float, K: float, T: float, r: float, sigma: float) -> Dict:
        """Perform sensitivity analysis using PEMC"""
        
        spot_range = np.linspace(S * 0.8, S * 1.2, 20)
        pemc_prices = []
        traditional_prices = []
        
        for spot in spot_range:
            pemc_result = self.pemc_estimate(spot, K, T, r, sigma, n_expensive=500, n_cheap=5000)
            traditional_result = self.black_scholes_price(spot, K, T, r, sigma)
            
            pemc_prices.append(pemc_result['pemc_price'])
            traditional_prices.append(traditional_result['call_price'])
        
        return {
            'spot_range': spot_range.tolist(),
            'pemc_prices': pemc_prices,
            'traditional_prices': traditional_prices
        }

# Example usage and testing
if __name__ == '__main__':
    print("=== PEMC Black-Scholes Options Pricing ===\n")
    
    # Initialize PEMC model
    pemc_model = PEMCBlackScholes()
    
    # Train the prediction model
    pemc_model.train_prediction_model(n_training_samples=50000)
    
    # Example parameters
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 0.25  # Time to expiration (3 months)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.20  # Volatility (20%)
    
    print(f"\nPricing call option with parameters:")
    print(f"Spot: ${S}, Strike: ${K}, Time: {T} years, Rate: {r*100}%, Vol: {sigma*100}%\n")
    
    # PEMC estimation
    pemc_results = pemc_model.pemc_estimate(S, K, T, r, sigma, n_expensive=2000, n_cheap=20000)
    
    print("=== Results Comparison ===")
    print(f"Traditional Black-Scholes: ${pemc_results['traditional_bs']:.4f}")
    print(f"Standard Monte Carlo:      ${pemc_results['monte_carlo']:.4f} ± ${pemc_results['mc_std']:.4f}")
    print(f"PEMC Enhanced:             ${pemc_results['pemc_price']:.4f} ± ${pemc_results['pemc_std']:.4f}")
    print(f"Variance Reduction:        {pemc_results['variance_reduction']:.1f}%")
    
    print(f"\nComputational Efficiency:")
    print(f"Expensive evaluations: {pemc_results['n_expensive']}")
    print(f"Cheap evaluations:     {pemc_results['n_cheap']}")
    print(f"Total ratio:           {pemc_results['n_cheap']/pemc_results['n_expensive']:.1f}:1")
