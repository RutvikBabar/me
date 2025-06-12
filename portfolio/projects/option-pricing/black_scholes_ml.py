import math
import json
from typing import Dict, List, Tuple

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S  # Spot price
        self.K = K  # Strike price
        self.T = T  # Time to maturity in years
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility
        # Removed dividend yield (q)

    def _N(self, x):
        """Cumulative distribution function for the standard normal distribution"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _pdf(self, x):
        """Probability density function for standard normal distribution"""
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

    def d1(self):
        return (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma * self.sigma) * self.T) / (self.sigma * math.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * math.sqrt(self.T)

    def call_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.S * self._N(d1) - self.K * math.exp(-self.r * self.T) * self._N(d2)

    def put_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.K * math.exp(-self.r * self.T) * self._N(-d2) - self.S * self._N(-d1)

    def delta(self):
        d1 = self.d1()
        call_delta = self._N(d1)
        put_delta = call_delta - 1
        return call_delta, put_delta

    def gamma(self):
        d1 = self.d1()
        pdf = self._pdf(d1)
        # Without dividend yield, no exponential term needed
        return pdf / (self.S * self.sigma * math.sqrt(self.T))


    def theta(self):
        d1 = self.d1()
        d2 = self.d2()
        pdf = self._pdf(d1)
        call_theta = (-self.S * pdf * self.sigma / (2 * math.sqrt(self.T))
                      - self.r * self.K * math.exp(-self.r * self.T) * self._N(d2)) / 365
        put_theta = (-self.S * pdf * self.sigma / (2 * math.sqrt(self.T))
                     + self.r * self.K * math.exp(-self.r * self.T) * self._N(-d2)) / 365
        return call_theta, put_theta

    def vega(self):
        d1 = self.d1()
        pdf = self._pdf(d1)
        return self.S * pdf * math.sqrt(self.T) / 100

    def rho(self):
        d2 = self.d2()
        call_rho = self.K * self.T * math.exp(-self.r * self.T) * self._N(d2) / 100
        put_rho = -self.K * self.T * math.exp(-self.r * self.T) * self._N(-d2) / 100
        return call_rho, put_rho

    def intrinsic_values(self):
        call_intrinsic = max(self.S - self.K, 0)
        put_intrinsic = max(self.K - self.S, 0)
        return call_intrinsic, put_intrinsic

    def time_values(self):
        call_intrinsic, put_intrinsic = self.intrinsic_values()
        call_time = self.call_price() - call_intrinsic
        put_time = self.put_price() - put_intrinsic
        return call_time, put_time

    def get_all_metrics(self):
        """Return all calculated metrics in a dictionary"""
        call_price = self.call_price()
        put_price = self.put_price()
        delta_call, delta_put = self.delta()
        gamma = self.gamma()
        theta_call, theta_put = self.theta()
        vega = self.vega()
        rho_call, rho_put = self.rho()
        call_intrinsic, put_intrinsic = self.intrinsic_values()
        call_time, put_time = self.time_values()

        return {
            'prices': {
                'call': call_price,
                'put': put_price,
                'call_intrinsic': call_intrinsic,
                'put_intrinsic': put_intrinsic,
                'call_time_value': call_time,
                'put_time_value': put_time
            },
            'greeks': {
                'delta_call': delta_call,
                'delta_put': delta_put,
                'gamma': gamma,
                'theta_call': theta_call,
                'theta_put': theta_put,
                'vega': vega,
                'rho_call': rho_call,
                'rho_put': rho_put
            }
        }

    def sensitivity_analysis(self, parameter: str, range_pct: float = 0.3, points: int = 50) -> Dict:
        """Perform sensitivity analysis on a specific parameter"""
        base_value = getattr(self, parameter)
        
        if parameter in ['r', 'sigma']:
            param_range = [max(0.001, base_value * (1 + i * range_pct / points)) for i in range(-points//2, points//2 + 1)]
        else:
            param_range = [base_value * (1 + i * range_pct / points) for i in range(-points//2, points//2 + 1)]
        
        results = {
            'parameter_values': param_range,
            'call_prices': [],
            'put_prices': [],
            'call_deltas': [],
            'put_deltas': [],
            'gammas': [],
            'vegas': []
        }
        
        for value in param_range:
            # Create temporary model with modified parameter
            temp_params = {
                'S': self.S, 'K': self.K, 'T': self.T, 
                'r': self.r, 'sigma': self.sigma
            }
            temp_params[parameter] = value
            
            temp_model = BlackScholesModel(**temp_params)
            
            results['call_prices'].append(temp_model.call_price())
            results['put_prices'].append(temp_model.put_price())
            
            delta_call, delta_put = temp_model.delta()
            results['call_deltas'].append(delta_call)
            results['put_deltas'].append(delta_put)
            results['gammas'].append(temp_model.gamma())
            results['vegas'].append(temp_model.vega())
        
        return results

# Example usage
if __name__ == '__main__':
    # Example calculation
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 0.25  # Time to expiration (3 months)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.20  # Volatility (20%)
    
    # Create Black-Scholes model
    bs_model = BlackScholesModel(S, K, T, r, sigma)
    
    # Calculate option prices
    print(f"Call Price: ${bs_model.call_price():.4f}")
    print(f"Put Price: ${bs_model.put_price():.4f}")
    
    # Calculate Greeks
    delta_call, delta_put = bs_model.delta()
    print(f"Call Delta: {delta_call:.3f}")
    print(f"Put Delta: {delta_put:.3f}")

class MLEnhancedBlackScholes(BlackScholesModel):
    def __init__(self, S, K, T, r, base_sigma):
        super().__init__(S, K, T, r, base_sigma)
        self.ml_volatility_predictor = self.load_volatility_model()
        self.market_regime_detector = self.load_regime_model()
        
    def adaptive_volatility(self, market_features):
        """Use ML to predict more accurate volatility"""
        # Features: VIX, historical volatility, market sentiment, etc.
        predicted_vol = self.ml_volatility_predictor.predict(market_features)
        
        # Detect market regime (bull/bear/sideways)
        regime = self.market_regime_detector.predict(market_features)
        
        # Adjust volatility based on regime
        regime_adjustments = {'bull': 0.9, 'bear': 1.2, 'sideways': 1.0}
        adjusted_vol = predicted_vol * regime_adjustments[regime]
        
        return adjusted_vol
    
    def ensemble_pricing(self, market_data):
        """Combine multiple models for robust pricing"""
        # Traditional Black-Scholes
        bs_price = self.call_price()
        
        # ML-enhanced volatility
        ml_vol = self.adaptive_volatility(market_data)
        ml_model = BlackScholesModel(self.S, self.K, self.T, self.r, ml_vol)
        ml_price = ml_model.call_price()
        
        # Monte Carlo with stochastic volatility
        mc_price = self.stochastic_vol_monte_carlo()
        
        # Weighted ensemble
        weights = [0.3, 0.5, 0.2]  # BS, ML, MC
        final_price = sum(w * p for w, p in zip(weights, [bs_price, ml_price, mc_price]))
        
        return {
            'ensemble_price': final_price,
            'traditional_bs': bs_price,
            'ml_enhanced': ml_price,
            'monte_carlo': mc_price,
            'confidence_interval': self.calculate_prediction_interval()
        }
