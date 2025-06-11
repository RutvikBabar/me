import math
import json
from typing import Dict, List, Tuple

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S  # Spot price
        self.K = K  # Strike price
        self.T = T  # Time to maturity in years
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility
        self.q = q  # Dividend yield

    def _N(self, x):
        """Cumulative distribution function for the standard normal distribution"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _pdf(self, x):
        """Probability density function for standard normal distribution"""
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

    def d1(self):
        return (math.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma * self.sigma) * self.T) / (self.sigma * math.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * math.sqrt(self.T)

    def call_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.S * math.exp(-self.q * self.T) * self._N(d1) - self.K * math.exp(-self.r * self.T) * self._N(d2)

    def put_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.K * math.exp(-self.r * self.T) * self._N(-d2) - self.S * math.exp(-self.q * self.T) * self._N(-d1)

    def delta(self):
        d1 = self.d1()
        call_delta = math.exp(-self.q * self.T) * self._N(d1)
        put_delta = call_delta - math.exp(-self.q * self.T)
        return call_delta, put_delta

    def gamma(self):
        d1 = self.d1()
        pdf = self._pdf(d1)
        return math.exp(-self.q * self.T) * pdf / (self.S * self.sigma * math.sqrt(self.T))

    def theta(self):
        d1 = self.d1()
        d2 = self.d2()
        pdf = self._pdf(d1)
        call_theta = (-self.S * pdf * self.sigma * math.exp(-self.q * self.T) / (2 * math.sqrt(self.T))
                      - self.r * self.K * math.exp(-self.r * self.T) * self._N(d2)
                      + self.q * self.S * math.exp(-self.q * self.T) * self._N(d1)) / 365
        put_theta = (-self.S * pdf * self.sigma * math.exp(-self.q * self.T) / (2 * math.sqrt(self.T))
                     + self.r * self.K * math.exp(-self.r * self.T) * self._N(-d2)
                     - self.q * self.S * math.exp(-self.q * self.T) * self._N(-d1)) / 365
        return call_theta, put_theta

    def vega(self):
        d1 = self.d1()
        pdf = self._pdf(d1)
        return self.S * math.exp(-self.q * self.T) * pdf * math.sqrt(self.T) / 100

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
        
        if parameter in ['r', 'sigma', 'q']:
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
                'r': self.r, 'sigma': self.sigma, 'q': self.q
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

class OptionsPortfolioAnalyzer:
    """Advanced analysis tools for options portfolio"""
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, r: float, 
                          option_type: str = 'call', q: float = 0, precision: float = 0.0001) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        sigma = 0.2  # Initial guess
        max_iterations = 100
        
        for i in range(max_iterations):
            model = BlackScholesModel(S, K, T, r, sigma, q)
            
            if option_type.lower() == 'call':
                price = model.call_price()
            else:
                price = model.put_price()
            
            vega = model.vega() * 100  # Convert back to percentage
            
            price_diff = price - market_price
            
            if abs(price_diff) < precision:
                return sigma
            
            if vega == 0:
                break
                
            sigma = sigma - price_diff / vega
            
            # Ensure sigma stays positive
            sigma = max(0.001, sigma)
        
        return sigma

    @staticmethod
    def monte_carlo_simulation(S: float, K: float, T: float, r: float, sigma: float, 
                              q: float = 0, num_simulations: int = 10000) -> Dict:
        """Monte Carlo simulation for option pricing"""
        import random
        
        dt = T / 252  # Daily time step
        call_payoffs = []
        put_payoffs = []
        
        for _ in range(num_simulations):
            St = S
            for _ in range(int(T * 252)):
                z = random.gauss(0, 1)
                St = St * math.exp((r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)
            
            call_payoffs.append(max(St - K, 0))
            put_payoffs.append(max(K - St, 0))
        
        discount_factor = math.exp(-r * T)
        
        return {
            'call_price': discount_factor * sum(call_payoffs) / num_simulations,
            'put_price': discount_factor * sum(put_payoffs) / num_simulations,
            'final_prices': [S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * random.gauss(0, 1)) for _ in range(100)]
        }
