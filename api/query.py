"""
Vercel Serverless Function for MADDUX Chat API.

Handles natural language queries about the MADDUX baseball analytics model.
User provides their own Anthropic API key with each request.
"""

from http.server import BaseHTTPRequestHandler
import json

# Try to import anthropic
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Top 2026 projections (from enhanced model)
TOP_2026_PROJECTIONS = [
    {"rank": 1, "player": "LaMonte Wade Jr.", "age": 31, "ops_2025": 0.524, "predicted_delta": 0.121, "projected_ops": 0.645, "key_factor": "High deviation from baseline (+0.25)"},
    {"rank": 2, "player": "Joc Pederson", "age": 33, "ops_2025": 0.614, "predicted_delta": 0.114, "projected_ops": 0.728, "key_factor": "Strong career peak deviation (+0.29)"},
    {"rank": 3, "player": "Henry Davis", "age": 25, "ops_2025": 0.512, "predicted_delta": 0.105, "projected_ops": 0.617, "key_factor": "Large underperformance gap (67)"},
    {"rank": 4, "player": "Anthony Santander", "age": 30, "ops_2025": 0.565, "predicted_delta": 0.092, "projected_ops": 0.657, "key_factor": "Deviation from baseline (+0.24)"},
    {"rank": 5, "player": "Tyler O'Neill", "age": 30, "ops_2025": 0.684, "predicted_delta": 0.091, "projected_ops": 0.775, "key_factor": "Career peak deviation (+0.23)"},
    {"rank": 6, "player": "Jose Iglesias", "age": 35, "ops_2025": 0.592, "predicted_delta": 0.090, "projected_ops": 0.682, "key_factor": "Underperformance vs expected stats"},
    {"rank": 7, "player": "Jordan Walker", "age": 23, "ops_2025": 0.584, "predicted_delta": 0.089, "projected_ops": 0.673, "key_factor": "Young age + high ceiling"},
    {"rank": 8, "player": "Randal Grichuk", "age": 33, "ops_2025": 0.674, "predicted_delta": 0.088, "projected_ops": 0.762, "key_factor": "Career peak deviation"},
    {"rank": 9, "player": "Nolan Jones", "age": 27, "ops_2025": 0.600, "predicted_delta": 0.082, "projected_ops": 0.682, "key_factor": "Underperformance gap"},
    {"rank": 10, "player": "Matt McLain", "age": 25, "ops_2025": 0.643, "predicted_delta": 0.079, "projected_ops": 0.722, "key_factor": "Deviation from baseline (+0.22)"},
]

# Walk-forward backtest results
BACKTEST_RESULTS = [
    {"year_pair": "2018→2019", "hit_rate": 85.0, "avg_ops_change": 0.111},
    {"year_pair": "2019→2020", "hit_rate": 80.0, "avg_ops_change": 0.055},
    {"year_pair": "2020→2021", "hit_rate": 85.0, "avg_ops_change": 0.101},
    {"year_pair": "2021→2022", "hit_rate": 60.0, "avg_ops_change": 0.019},
    {"year_pair": "2022→2023", "hit_rate": 95.0, "avg_ops_change": 0.101},
    {"year_pair": "2023→2024", "hit_rate": 65.0, "avg_ops_change": 0.024},
    {"year_pair": "2024→2025", "hit_rate": 85.0, "avg_ops_change": 0.071},
]


def get_context():
    """Generate context about the model for Claude."""
    
    projections_text = "\n".join([
        f"  {p['rank']}. {p['player']} (Age {p['age']}): "
        f"2025 OPS {p['ops_2025']:.3f} → Predicted +{p['predicted_delta']:.3f} → "
        f"2026 OPS {p['projected_ops']:.3f} | Factor: {p['key_factor']}"
        for p in TOP_2026_PROJECTIONS
    ])
    
    backtest_text = "\n".join([
        f"  {b['year_pair']}: {b['hit_rate']:.0f}% hit rate, avg OPS change +{b['avg_ops_change']:.3f}"
        for b in BACKTEST_RESULTS
    ])
    
    return f"""
MADDUX™ ANALYTICS SYSTEM
========================
AutoCoach LLC | Phase 1 Validated Model

MODEL ARCHITECTURE:
- Stacking Meta-Learner ensemble (Ridge, Lasso, Gradient Boosting)
- 25+ engineered features across 3 categories
- Walk-forward cross-validation (no data leakage)

KEY FEATURES (by importance):
1. deviation_from_baseline (r=0.49): Current OPS vs career expected baseline
2. improvement_momentum (r=-0.46): NEGATIVE predictor - recent improvers tend to REGRESS
3. career_peak_deviation (r=0.38): Distance from personal career best
4. underperformance_gap (r=0.35): xwOBA minus wOBA (luck adjustment)
5. combined_luck_index (r=0.33): BABIP and ISO luck indicators

VALIDATED PERFORMANCE:
- Correlation: 0.50
- R-squared: 28%
- Walk-Forward Hit Rate: 79.3%

BACKTEST RESULTS:
{backtest_text}

TOP 2026 BREAKOUT CANDIDATES:
{projections_text}

KEY INSIGHTS:
- Original MADDUX formula had NEGATIVE correlation (rejected)
- Players who improved recently tend to REGRESS (improvement_momentum is negative)
- Underperformers (xwOBA > wOBA) bounce back - this is why LaMonte Wade Jr. ranks #1
- Young players with high deviation from baseline have most upside

WHY LAMONTE WADE JR. (#1):
- 2025 OPS: 0.524 (very low for his talent level)
- Deviation from baseline: +0.25 (performing well below expected)
- The model predicts +0.121 OPS improvement to 0.645
- Key factors: High underperformance gap, deviation from career baseline
- He's "due" for positive regression based on underlying metrics
"""


def execute_query(question: str, api_key: str) -> dict:
    """Execute a query using Claude with the user's API key."""
    
    if not ANTHROPIC_AVAILABLE:
        return {"error": "Anthropic library not available on server"}
    
    context = get_context()
    
    system_prompt = """You are a baseball analytics expert for AutoCoach LLC, helping users understand the MADDUX™ predictive model.

Your role:
1. Answer questions about player projections, model methodology, and accuracy
2. Be specific with player names, numbers, and statistical details
3. Explain reasoning using feature importance data
4. Note that "improvement momentum" is NEGATIVE - recent improvers regress
5. When asked about specific players, reference their key factors from the data

Keep responses concise but informative (2-3 paragraphs max)."""

    user_prompt = f"""{context}

USER QUESTION: {question}

Provide a helpful, data-driven response based on the MADDUX model context above."""

    try:
        client = Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        return {"answer": message.content[0].text}
        
    except Exception as e:
        error_str = str(e)
        if "authentication" in error_str.lower() or "invalid" in error_str.lower():
            return {"error": "Invalid API key. Please check your Anthropic API key."}
        elif "rate" in error_str.lower():
            return {"error": "Rate limit exceeded. Please wait a moment and try again."}
        return {"error": f"API Error: {error_str}"}


class handler(BaseHTTPRequestHandler):
    """Vercel serverless handler."""
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', '0')
        self.end_headers()
    
    def do_GET(self):
        """Handle health check."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {"status": "healthy", "service": "MADDUX Chat API"}
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        """Handle query requests."""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            question = data.get('question', '')
            api_key = data.get('api_key', '')
            
            # Validate API key
            if not api_key or not api_key.startswith('sk-ant-'):
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Invalid API key format. Key should start with 'sk-ant-'"
                }).encode())
                return
            
            # Validate question
            if not question or len(question.strip()) < 3:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Question is too short"
                }).encode())
                return
            
            # Execute query
            result = execute_query(question.strip(), api_key)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Invalid JSON in request body"
            }).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": f"Server error: {str(e)}"
            }).encode())
