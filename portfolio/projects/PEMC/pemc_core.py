import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import json
import time

@dataclass
class PEMCResult:
    """Results from PEMC estimation"""
    pemc_estimate: float
    traditional_mc: float
    variance_reduction: float
    confidence_interval: Tuple[float, float]
    n_expensive: int
    n_cheap: int
    ml_confidence: float
    correlation: float
    cost_ratio: float

class PEMCBlackScholes:
    """
    Complete PEMC-Enhanced Black-Scholes Implementation
    Following Li et al. (2024) arXiv:2412.11257v3
    """
    
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
    
    def simulate_gbm_path(self, S0: float, r: float, sigma: float, T: float, dt: float = 1/252) -> Tuple[np.ndarray, float]:
        """
        Simulate GBM path and return (price_path, brownian_sum)
        Returns both the full path and the sum of Brownian increments (feature X)
        """
        n_steps = int(T / dt)
        prices = np.zeros(n_steps + 1)
        prices[0] = S0
        
        brownian_increments = np.random.normal(0, math.sqrt(dt), n_steps)
        brownian_sum = np.sum(brownian_increments)  # This is our feature X
        
        for i in range(n_steps):
            prices[i + 1] = prices[i] * math.exp((r - 0.5 * sigma**2) * dt + sigma * brownian_increments[i])
        
        return prices, brownian_sum
    
    def asian_option_payoff(self, price_path: np.ndarray, K: float) -> float:
        """Calculate Asian option payoff"""
        arithmetic_average = np.mean(price_path[1:])  # Exclude initial price
        return max(arithmetic_average - K, 0)
    
    def generate_training_data(self, n_samples: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for PEMC predictor"""
        print(f"Generating {n_samples} training samples...")
        
        features = []
        labels = []
        
        # Parameter ranges for training
        S_range = (80, 120)
        K_range = (90, 110)
        T_range = (0.1, 1.0)
        r_range = (0.01, 0.08)
        sigma_range = (0.1, 0.4)
        
        for i in range(n_samples):
            if i % 5000 == 0:
                print(f"Generated {i}/{n_samples} samples")
            
            # Sample parameters
            S = random.uniform(*S_range)
            K = random.uniform(*K_range)
            T = random.uniform(*T_range)
            r = random.uniform(*r_range)
            sigma = random.uniform(*sigma_range)
            
            # Simulate path and get feature
            price_path, brownian_sum = self.simulate_gbm_path(S, r, sigma, T)
            
            # Calculate payoff
            payoff = self.asian_option_payoff(price_path, K)
            
            # Create feature vector: [S, K, T, r, sigma, brownian_sum]
            feature_vector = [S, K, T, r, sigma, brownian_sum]
            features.append(feature_vector)
            labels.append(payoff)
        
        return np.array(features), np.array(labels)
    
    def train_prediction_model(self, n_training_samples: int = 50000):
        """Train the ML prediction model for PEMC"""
        print("Training PEMC prediction model...")
        
        # Generate training data
        X_train, y_train = self.generate_training_data(n_training_samples)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create neural network
        self.prediction_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(6,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        self.prediction_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        history = self.prediction_model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=256,
            validation_split=0.2,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        self.is_trained = True
        
        # Calculate training accuracy
        y_pred = self.prediction_model.predict(X_train_scaled, verbose=0)
        mse = np.mean((y_train - y_pred.flatten())**2)
        print(f"Training completed. MSE: {mse:.6f}")
        
        return mse
    
    def pemc_estimate(self, S: float, K: float, T: float, r: float, sigma: float, 
                     n_expensive: int = 1000, n_cheap: int = 10000) -> PEMCResult:
        """
        PEMC estimation following Equation (2) from the paper:
        PEMC = (1/n)Σ[f(Yi) - g(Xi)] + (1/N)Σ[g(X̃j)]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before estimation")
        
        # Step 1: Generate n expensive coupled samples (Yi, Xi)
        expensive_samples = []
        
        for i in range(n_expensive):
            # Generate coupled (Y, X) sample
            price_path, brownian_sum = self.simulate_gbm_path(S, r, sigma, T)
            true_payoff = self.asian_option_payoff(price_path, K)
            
            # ML prediction g(Xi)
            feature_vector = np.array([[S, K, T, r, sigma, brownian_sum]])
            feature_scaled = self.scaler.transform(feature_vector)
            ml_prediction = self.prediction_model.predict(feature_scaled, verbose=0)[0, 0]
            
            # Store f(Yi) - g(Xi)
            expensive_samples.append(true_payoff - ml_prediction)
        
        # Step 2: Generate N cheap independent samples X̃j
        cheap_predictions = []
        
        for j in range(n_cheap):
            # Generate independent Brownian sum with same distribution
            dt = T / 252
            n_steps = int(T / dt)
            brownian_increments = np.random.normal(0, math.sqrt(dt), n_steps)
            brownian_sum = np.sum(brownian_increments)
            
            # ML prediction g(X̃j)
            feature_vector = np.array([[S, K, T, r, sigma, brownian_sum]])
            feature_scaled = self.scaler.transform(feature_vector)
            ml_prediction = self.prediction_model.predict(feature_scaled, verbose=0)[0, 0]
            
            cheap_predictions.append(ml_prediction)
        
        # Step 3: PEMC estimator (Equation 2)
        expensive_term = np.mean(expensive_samples)
        cheap_term = np.mean(cheap_predictions)
        pemc_price = expensive_term + cheap_term
        
        # Calculate traditional MC for comparison
        traditional_mc = self._traditional_monte_carlo(S, K, T, r, sigma, n_expensive)
        
        # Calculate variance components
        expensive_var = np.var(expensive_samples) / n_expensive
        cheap_var = np.var(cheap_predictions) / n_cheap
        pemc_var = expensive_var + cheap_var
        pemc_std = math.sqrt(pemc_var)
        
        # Traditional MC variance
        mc_samples = []
        for _ in range(n_expensive):
            price_path, _ = self.simulate_gbm_path(S, r, sigma, T)
            payoff = self.asian_option_payoff(price_path, K)
            mc_samples.append(payoff)
        
        mc_var = np.var(mc_samples) / n_expensive
        mc_std = math.sqrt(mc_var)
        
        # Variance reduction calculation
        variance_reduction = (mc_var - pemc_var) / mc_var * 100 if mc_var > 0 else 0
        
        # Calculate correlation and cost ratio
        correlation = self._calculate_correlation(S, K, T, r, sigma, 1000)
        cost_ratio = 0.001  # Cheap samples are 1000x cheaper
        
        # Confidence interval
        z_alpha = 1.96  # 95% confidence
        margin = z_alpha * pemc_std
        confidence_interval = (pemc_price - margin, pemc_price + margin)
        
        return PEMCResult(
            pemc_estimate=pemc_price,
            traditional_mc=traditional_mc,
            variance_reduction=variance_reduction,
            confidence_interval=confidence_interval,
            n_expensive=n_expensive,
            n_cheap=n_cheap,
            ml_confidence=min(0.95, 0.7 + correlation * 0.3),
            correlation=correlation,
            cost_ratio=cost_ratio
        )
    
    def _traditional_monte_carlo(self, S: float, K: float, T: float, r: float, sigma: float, n_samples: int) -> float:
        """Calculate traditional Monte Carlo estimate"""
        samples = []
        for _ in range(n_samples):
            price_path, _ = self.simulate_gbm_path(S, r, sigma, T)
            payoff = self.asian_option_payoff(price_path, K)
            samples.append(payoff)
        return np.mean(samples)
    
    def _calculate_correlation(self, S: float, K: float, T: float, r: float, sigma: float, n_samples: int = 1000) -> float:
        """Calculate correlation between true payoffs and ML predictions"""
        true_payoffs = []
        ml_predictions = []
        
        for _ in range(n_samples):
            price_path, brownian_sum = self.simulate_gbm_path(S, r, sigma, T)
            true_payoff = self.asian_option_payoff(price_path, K)
            
            feature_vector = np.array([[S, K, T, r, sigma, brownian_sum]])
            feature_scaled = self.scaler.transform(feature_vector)
            ml_prediction = self.prediction_model.predict(feature_scaled, verbose=0)[0, 0]
            
            true_payoffs.append(true_payoff)
            ml_predictions.append(ml_prediction)
        
        correlation = np.corrcoef(true_payoffs, ml_predictions)[0, 1]
        return max(0, min(1, correlation))  # Bound between 0 and 1
    
    def sensitivity_analysis(self, S: float, K: float, T: float, r: float, sigma: float, 
                           parameter: str = 'S', range_pct: float = 0.3, points: int = 20) -> Dict:
        """Perform sensitivity analysis varying a specific parameter"""
        param_map = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
        base_value = param_map[parameter]
        
        if parameter in ['r', 'sigma']:
            param_range = np.linspace(max(0.001, base_value * (1 - range_pct)), 
                                    base_value * (1 + range_pct), points)
        else:
            param_range = np.linspace(base_value * (1 - range_pct), 
                                    base_value * (1 + range_pct), points)
        
        results = {
            'parameter_values': param_range.tolist(),
            'pemc_prices': [],
            'traditional_prices': [],
            'variance_reductions': []
        }
        
        for value in param_range:
            temp_params = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}
            temp_params[parameter] = value
            
            pemc_result = self.pemc_estimate(**temp_params, n_expensive=500, n_cheap=5000)
            
            results['pemc_prices'].append(pemc_result.pemc_estimate)
            results['traditional_prices'].append(pemc_result.traditional_mc)
            results['variance_reductions'].append(pemc_result.variance_reduction)
        
        return results

# Initialize and train the model
pemc_model = PEMCBlackScholes()
