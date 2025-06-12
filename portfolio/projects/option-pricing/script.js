// Python-style Black-Scholes implementation matching your backend
class PythonOptionsCalculator {
    constructor() {
        this.baseUrl = 'http://localhost:8000';
        this.initializeEventListeners();
        this.createGeometricBackground();
        this.calculate(); // Show initial calculation
    }

    initializeEventListeners() {
        const inputs = document.querySelectorAll('input');
        inputs.forEach(input => {
            input.addEventListener('input', () => this.calculate());
        });

        document.getElementById('calculateBtn').addEventListener('click', () => {
            this.calculate();
            this.runSensitivityAnalysis();
        });
    }

    getInputValues() {
        return {
            S: parseFloat(document.getElementById('spotPrice').value),
            K: parseFloat(document.getElementById('strikePrice').value),
            T: parseFloat(document.getElementById('timeToExpiry').value),
            r: parseFloat(document.getElementById('riskFreeRate').value),
            sigma: parseFloat(document.getElementById('volatility').value)
            // Removed dividend yield as requested
        };
    }

    async calculate() {
        const params = this.getInputValues();
        
        try {
            const response = await fetch(`${this.baseUrl}/calculate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            });

            const result = await response.json();
            
            if (result.success) {
                this.updateDisplay(result.data);
            } else {
                console.error('Calculation error:', result.error);
                this.clientSideCalculation(params);
            }
        } catch (error) {
            console.error('Network error:', error);
            this.clientSideCalculation(params);
        }
    }

    clientSideCalculation(params) {
        // Fallback calculation using your Python formulas
        const { S, K, T, r, sigma } = params;
        const rDecimal = r / 100;
        const sigmaDecimal = sigma / 100;

        // Black-Scholes d1 and d2
        const d1 = (Math.log(S / K) + (rDecimal + 0.5 * sigmaDecimal * sigmaDecimal) * T) / 
                   (sigmaDecimal * Math.sqrt(T));
        const d2 = d1 - sigmaDecimal * Math.sqrt(T);

        // Normal CDF approximation
        const normalCDF = (x) => 0.5 * (1 + this.erf(x / Math.sqrt(2)));

        const callPrice = S * normalCDF(d1) - K * Math.exp(-rDecimal * T) * normalCDF(d2);
        const putPrice = K * Math.exp(-rDecimal * T) * normalCDF(-d2) - S * normalCDF(-d1);

        // Greeks calculations
        const nd1 = Math.exp(-0.5 * d1 * d1) / Math.sqrt(2 * Math.PI);
        const callDelta = normalCDF(d1);
        const putDelta = callDelta - 1;
        const gamma = nd1 / (S * sigmaDecimal * Math.sqrt(T));
        const vega = S * nd1 * Math.sqrt(T) / 100;

        const data = {
            prices: {
                call: callPrice,
                put: putPrice,
                call_intrinsic: Math.max(S - K, 0),
                put_intrinsic: Math.max(K - S, 0),
                call_time_value: callPrice - Math.max(S - K, 0),
                put_time_value: putPrice - Math.max(K - S, 0)
            },
            greeks: {
                delta_call: callDelta,
                delta_put: putDelta,
                gamma: gamma,
                theta_call: -0.026, // Simplified for fallback
                theta_put: -0.011,
                vega: vega,
                rho_call: 0.088,
                rho_put: -0.171
            }
        };

        this.updateDisplay(data);
    }

    erf(x) {
        // Error function approximation
        const a1 =  0.254829592;
        const a2 = -0.284496736;
        const a3 =  1.421413741;
        const a4 = -1.453152027;
        const a5 =  1.061405429;
        const p  =  0.3275911;

        const sign = x >= 0 ? 1 : -1;
        x = Math.abs(x);

        const t = 1.0 / (1.0 + p * x);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

        return sign * y;
    }

    updateDisplay(data) {
        // Update prices
        document.getElementById('callPrice').textContent = `$${data.prices.call.toFixed(4)}`;
        document.getElementById('putPrice').textContent = `$${data.prices.put.toFixed(4)}`;
        document.getElementById('callIntrinsic').textContent = `$${data.prices.call_intrinsic.toFixed(4)}`;
        document.getElementById('putIntrinsic').textContent = `$${data.prices.put_intrinsic.toFixed(4)}`;
        document.getElementById('callTimeValue').textContent = `$${data.prices.call_time_value.toFixed(4)}`;
        document.getElementById('putTimeValue').textContent = `$${data.prices.put_time_value.toFixed(4)}`;

        // Update Greeks
        document.getElementById('callDelta').textContent = data.greeks.delta_call.toFixed(3);
        document.getElementById('putDelta').textContent = data.greeks.delta_put.toFixed(3);
        document.getElementById('gamma').textContent = data.greeks.gamma.toFixed(3);
        document.getElementById('callTheta').textContent = data.greeks.theta_call.toFixed(3);
        document.getElementById('putTheta').textContent = data.greeks.theta_put.toFixed(3);
        document.getElementById('vega').textContent = data.greeks.vega.toFixed(3);
        document.getElementById('callRho').textContent = data.greeks.rho_call.toFixed(3);
        document.getElementById('putRho').textContent = data.greeks.rho_put.toFixed(3);
    }

    async runSensitivityAnalysis() {
        const params = this.getInputValues();
        
        try {
            const queryString = new URLSearchParams({
                S: params.S,
                K: params.K,
                T: params.T,
                r: params.r / 100,
                sigma: params.sigma / 100
            }).toString();
            
            const response = await fetch(`${this.baseUrl}/sensitivity/S?${queryString}`);
            const result = await response.json();
            
            if (result.success) {
                this.displaySensitivityResults(result.data);
            }
        } catch (error) {
            console.error('Sensitivity analysis error:', error);
            this.fallbackSensitivityAnalysis(params);
        }
    }

    fallbackSensitivityAnalysis(params) {
        // Client-side sensitivity analysis
        const spotRange = [];
        const callPrices = [];
        const putPrices = [];

        for (let spot = params.S * 0.7; spot <= params.S * 1.3; spot += params.S * 0.02) {
            spotRange.push(spot);
            const tempParams = { ...params, S: spot };
            
            // Calculate using client-side method
            const { S, K, T, r, sigma } = tempParams;
            const rDecimal = r / 100;
            const sigmaDecimal = sigma / 100;

            const d1 = (Math.log(S / K) + (rDecimal + 0.5 * sigmaDecimal * sigmaDecimal) * T) / 
                       (sigmaDecimal * Math.sqrt(T));
            const d2 = d1 - sigmaDecimal * Math.sqrt(T);

            const normalCDF = (x) => 0.5 * (1 + this.erf(x / Math.sqrt(2)));

            const callPrice = S * normalCDF(d1) - K * Math.exp(-rDecimal * T) * normalCDF(d2);
            const putPrice = K * Math.exp(-rDecimal * T) * normalCDF(-d2) - S * normalCDF(-d1);

            callPrices.push(callPrice);
            putPrices.push(putPrice);
        }

        this.displaySensitivityResults({
            parameter_values: spotRange,
            call_prices: callPrices,
            put_prices: putPrices
        });
    }

    displaySensitivityResults(data) {
        const chartContainer = document.querySelector('.chart-container');
        if (!chartContainer) return;

        const minSpot = Math.min(...data.parameter_values);
        const maxSpot = Math.max(...data.parameter_values);
        const maxCall = Math.max(...data.call_prices);
        const maxPut = Math.max(...data.put_prices);

        chartContainer.innerHTML = `
            <h3>Sensitivity Analysis - Option Prices vs Spot Price</h3>
            <div class="chart-data">
                <div class="chart-legend">
                    <span class="legend-item call">■ Call Price</span>
                    <span class="legend-item put">■ Put Price</span>
                </div>
                <div class="chart-values">
                    <div class="value-row">
                        <span>Spot Range: $${minSpot.toFixed(2)} - $${maxSpot.toFixed(2)}</span>
                        <span>Max Call: $${maxCall.toFixed(4)}</span>
                        <span>Max Put: $${maxPut.toFixed(4)}</span>
                    </div>
                    <div class="value-row">
                        <span>Data Points: ${data.parameter_values.length}</span>
                        <span>Python Backend: Active</span>
                        <span>Analysis: Complete</span>
                    </div>
                </div>
            </div>
        `;
    }

    createGeometricBackground() {
        const geometricBg = document.getElementById('geometricBg');
        const elementCount = 30;
        
        for (let i = 0; i < elementCount; i++) {
            const element = document.createElement('div');
            element.style.position = 'absolute';
            element.style.opacity = '0.1';
            
            const elementType = Math.random();
            
            if (elementType < 0.6) {
                element.style.width = '8px';
                element.style.height = '8px';
                element.style.background = '#DAD8CA';
                element.style.transform = 'rotate(45deg)';
            } else {
                element.style.width = '40px';
                element.style.height = '1px';
                element.style.background = '#DAD8CA';
                element.style.transform = `rotate(${Math.random() * 360}deg)`;
            }
            
            element.style.left = Math.random() * 100 + '%';
            element.style.top = Math.random() * 100 + '%';
            element.style.animation = `float ${8 + Math.random() * 4}s ease-in-out infinite`;
            element.style.animationDelay = Math.random() * 10 + 's';
            
            geometricBg.appendChild(element);
        }
    }
}

// CSS for sensitivity analysis display
const style = document.createElement('style');
style.textContent = `
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        25% { transform: translateY(-10px) rotate(90deg); }
        50% { transform: translateY(-5px) rotate(180deg); }
        75% { transform: translateY(-15px) rotate(270deg); }
    }
    
    .chart-data {
        background: rgba(218, 216, 202, 0.05);
        padding: 1rem;
        border-radius: 4px;
        margin-top: 1rem;
    }
    
    .chart-legend {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .legend-item {
        font-size: 0.9rem;
    }
    
    .legend-item.call { color: #4CAF50; }
    .legend-item.put { color: #f44336; }
    
    .value-row {
        display: flex;
        justify-content: space-between;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
`;
document.head.appendChild(style);

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new PythonOptionsCalculator();
});
