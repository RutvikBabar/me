// PEMC Options Pricing System
class PEMCSystem {
    constructor() {
        this.currentChart = null;
        this.modelReady = false;
        this.initializeSystem();
        this.createMatrixEffect();
        this.checkModelStatus();
    }

    initializeSystem() {
        // Initialize event listeners
        document.getElementById('runPEMC').addEventListener('click', () => {
            this.executePEMC();
        });

        // Real-time parameter updates
        const inputs = document.querySelectorAll('input[type="number"]');
        inputs.forEach(input => {
            input.addEventListener('input', () => {
                if (this.modelReady) {
                    this.updateCalculations();
                }
            });
        });
    }

    async checkModelStatus() {
        try {
            const response = await fetch('/api/model_status');
            const status = await response.json();
            
            this.updateTrainingProgress(status);
            
            if (status.status === 'completed') {
                this.modelReady = true;
                this.showMainInterface();
                this.updateCalculations(); // Initial calculation
            } else if (status.status === 'training' || status.status === 'not_started') {
                // Check again in 2 seconds
                setTimeout(() => this.checkModelStatus(), 2000);
            } else if (status.status === 'error') {
                this.showError(status.error);
            }
        } catch (error) {
            console.error('Error checking model status:', error);
            setTimeout(() => this.checkModelStatus(), 5000);
        }
    }

    updateTrainingProgress(status) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const statusIndicator = document.getElementById('statusIndicator');
        const systemStatus = document.getElementById('systemStatus');

        if (status.status === 'training') {
            progressFill.style.width = `${status.progress}%`;
            progressText.textContent = `Training neural network... ${status.progress}%`;
            statusIndicator.className = 'status-indicator training';
            systemStatus.textContent = 'TRAINING MODEL';
        } else if (status.status === 'completed') {
            progressFill.style.width = '100%';
            progressText.textContent = 'Training completed! Model ready.';
            statusIndicator.className = 'status-indicator active';
            systemStatus.textContent = 'SYSTEM ONLINE';
        } else if (status.status === 'error') {
            progressText.textContent = `Error: ${status.error}`;
            statusIndicator.className = 'status-indicator error';
            systemStatus.textContent = 'SYSTEM ERROR';
        }
    }

    showMainInterface() {
        document.getElementById('trainingSection').style.display = 'none';
        document.getElementById('mainInterface').style.display = 'grid';
        document.getElementById('analyticsSection').style.display = 'grid';
    }

    showError(error) {
        const progressText = document.getElementById('progressText');
        progressText.textContent = `Error: ${error}`;
        progressText.style.color = '#f44336';
    }

    async updateCalculations() {
        if (!this.modelReady) return;

        const S = parseFloat(document.getElementById('spotPrice').value);
        const K = parseFloat(document.getElementById('strikePrice').value);
        const T = parseFloat(document.getElementById('timeToExpiry').value);
        const r = parseFloat(document.getElementById('riskFreeRate').value);
        const sigma = parseFloat(document.getElementById('volatility').value);

        if (isNaN(S) || isNaN(K) || isNaN(T) || isNaN(r) || isNaN(sigma)) return;

        try {
            const response = await fetch('/api/calculate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    S: S,
                    K: K,
                    T: T,
                    r: r,
                    sigma: sigma,
                    n_expensive: 500,  // Smaller for real-time updates
                    n_cheap: 5000
                })
            });

            const result = await response.json();

            if (result.success) {
                this.updateDisplay(result);
            } else {
                console.error('Calculation error:', result.error);
            }
        } catch (error) {
            console.error('Network error:', error);
        }
    }

    updateDisplay(data) {
        // Update traditional pricing
        document.getElementById('traditionalPrice').textContent = `$${data.traditional.call_price.toFixed(4)}`;
        document.getElementById('intrinsicValue').textContent = `$${data.traditional.call_intrinsic.toFixed(4)}`;
        document.getElementById('timeValue').textContent = `$${data.traditional.call_time_value.toFixed(4)}`;

        // Update PEMC pricing
        document.getElementById('traditionalMC').textContent = `$${data.pemc.traditional_mc.toFixed(4)}`;
        document.getElementById('pemcEstimate').textContent = `$${data.pemc.call_price.toFixed(4)}`;
        document.getElementById('pemcPrice').textContent = `$${data.pemc.call_price.toFixed(4)}`;
        
        const ci = data.pemc.confidence_interval;
        document.getElementById('confidenceInterval').textContent = `$${ci[0].toFixed(4)} - $${ci[1].toFixed(4)}`;

        // Update Greeks
        const greeks = data.traditional.greeks;
        document.getElementById('callDelta').textContent = greeks.call_delta.toFixed(3);
        document.getElementById('putDelta').textContent = greeks.put_delta.toFixed(3);
        document.getElementById('gamma').textContent = greeks.gamma.toFixed(3);
        document.getElementById('callTheta').textContent = greeks.call_theta.toFixed(3);
        document.getElementById('putTheta').textContent = greeks.put_theta.toFixed(3);
        document.getElementById('vega').textContent = greeks.vega.toFixed(3);
        document.getElementById('callRho').textContent = greeks.call_rho.toFixed(3);
        document.getElementById('putRho').textContent = greeks.put_rho.toFixed(3);

        // Update PEMC metrics
        document.getElementById('varianceReduction').textContent = `${data.pemc.variance_reduction.toFixed(1)}%`;
        document.getElementById('mlConfidence').textContent = `${(data.pemc.ml_confidence * 100).toFixed(0)}%`;
        document.getElementById('correlation').textContent = data.pemc.correlation.toFixed(3);
        document.getElementById('costRatio').textContent = data.pemc.cost_ratio.toFixed(3);
    }

    async executePEMC() {
        if (!this.modelReady) return;

        const S = parseFloat(document.getElementById('spotPrice').value);
        const K = parseFloat(document.getElementById('strikePrice').value);
        const T = parseFloat(document.getElementById('timeToExpiry').value);
        const r = parseFloat(document.getElementById('riskFreeRate').value);
        const sigma = parseFloat(document.getElementById('volatility').value);
        const nExpensive = parseInt(document.getElementById('nExpensive').value);
        const nCheap = parseInt(document.getElementById('nCheap').value);

        // Show execution state
        const btn = document.getElementById('runPEMC');
        btn.innerHTML = '<span>EXECUTING...</span><div class="btn-line"></div>';
        btn.disabled = true;

        try {
            const response = await fetch('/api/calculate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    S: S,
                    K: K,
                    T: T,
                    r: r,
                    sigma: sigma,
                    n_expensive: nExpensive,
                    n_cheap: nCheap
                })
            });

            const result = await response.json();

            if (result.success) {
                this.updateDisplay(result);
                await this.runSensitivityAnalysis();
            } else {
                console.error('PEMC execution error:', result.error);
            }
        } catch (error) {
            console.error('Network error:', error);
        } finally {
            // Reset button
            btn.innerHTML = '<span>EXECUTE PEMC</span><div class="btn-line"></div>';
            btn.disabled = false;
        }
    }

    async runSensitivityAnalysis() {
        const S = parseFloat(document.getElementById('spotPrice').value);
        const K = parseFloat(document.getElementById('strikePrice').value);
        const T = parseFloat(document.getElementById('timeToExpiry').value);
        const r = parseFloat(document.getElementById('riskFreeRate').value);
        const sigma = parseFloat(document.getElementById('volatility').value);

        try {
            const response = await fetch('/api/sensitivity', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    S: S,
                    K: K,
                    T: T,
                    r: r,
                    sigma: sigma,
                    parameter: 'S'
                })
            });

            const result = await response.json();

            if (result.success) {
                this.updateChart(result.data);
            }
        } catch (error) {
            console.error('Sensitivity analysis error:', error);
        }
    }

    updateChart(data) {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        
        if (this.currentChart) {
            this.currentChart.destroy();
        }

        this.currentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.parameter_values.map(v => v.toFixed(1)),
                datasets: [{
                    label: 'PEMC Enhanced',
                    data: data.pemc_prices,
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderWidth: 3,
                    fill: false
                }, {
                    label: 'Traditional MC',
                    data: data.traditional_prices,
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    borderWidth: 2,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: {
                            color: '#DAD8CA',
                            font: {
                                family: 'Orbitron'
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Spot Price',
                            color: '#DAD8CA'
                        },
                        ticks: {
                            color: '#DAD8CA',
                            font: {
                                family: 'Orbitron'
                            }
                        },
                        grid: {
                            color: 'rgba(218, 216, 202, 0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Option Price',
                            color: '#DAD8CA'
                        },
                        ticks: {
                            color: '#DAD8CA',
                            font: {
                                family: 'Orbitron'
                            }
                        },
                        grid: {
                            color: 'rgba(218, 216, 202, 0.1)'
                        }
                    }
                }
            }
        });
    }

    createMatrixEffect() {
        const matrixBg = document.getElementById('matrixBg');
        const elementCount = 50;
        
        for (let i = 0; i < elementCount; i++) {
            const element = document.createElement('div');
            element.style.position = 'absolute';
            element.style.width = '2px';
            element.style.height = '20px';
            element.style.background = 'rgba(218, 216, 202, 0.1)';
            element.style.left = Math.random() * 100 + '%';
            element.style.top = Math.random() * 100 + '%';
            element.style.animation = `matrixFall ${3 + Math.random() * 4}s linear infinite`;
            element.style.animationDelay = Math.random() * 5 + 's';
            
            matrixBg.appendChild(element);
        }
    }
}

// Chart management functions
function showChart(type) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    console.log(`Showing ${type} chart`);
}

// CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes matrixFall {
        0% { transform: translateY(-20px); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateY(100vh); opacity: 0; }
    }
    
    .status-indicator.training {
        background: #FF9800;
        animation: pulse 1s infinite;
    }
    
    .status-indicator.error {
        background: #f44336;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
`;
document.head.appendChild(style);

// Initialize system
document.addEventListener('DOMContentLoaded', () => {
    new PEMCSystem();
});
