from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import urllib.parse
import os
from black_scholes import BlackScholesModel, OptionsPortfolioAnalyzer

class OptionsHandler(SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        # Handle CORS preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/calculate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                # Remove dividend yield as requested
                model = BlackScholesModel(
                    S=data['S'],
                    K=data['K'],
                    T=data['T'],
                    r=data['r'] / 100,  # Convert percentage to decimal
                    sigma=data['sigma'] / 100,  # Convert percentage to decimal
                    q=0  # No dividend yield
                )
                
                results = model.get_all_metrics()
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = json.dumps({'success': True, 'data': results})
                self.wfile.write(response.encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = json.dumps({'success': False, 'error': str(e)})
                self.wfile.write(response.encode())
    
    def do_GET(self):
        if self.path.startswith('/sensitivity'):
            try:
                # Parse URL parameters
                url_parts = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(url_parts.query)
                
                # Extract parameter name from path
                parameter = url_parts.path.split('/')[-1]
                
                model = BlackScholesModel(
                    S=float(params['S'][0]),
                    K=float(params['K'][0]),
                    T=float(params['T'][0]),
                    r=float(params['r'][0]),
                    sigma=float(params['sigma'][0]),
                    q=0  # No dividend yield
                )
                
                results = model.sensitivity_analysis(parameter)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = json.dumps({'success': True, 'data': results})
                self.wfile.write(response.encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = json.dumps({'success': False, 'error': str(e)})
                self.wfile.write(response.encode())
        else:
            # Serve static files
            super().do_GET()

    def end_headers(self):
        # Add CORS headers to all responses
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == '__main__':
    # Change directory to serve static files from the correct location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    server = HTTPServer(('localhost', 8000), OptionsHandler)
    print("Python Options Pricing Server running on http://localhost:8000")
    print("Serving files from:", os.getcwd())
    server.serve_forever()
