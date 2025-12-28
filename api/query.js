// Vercel Serverless Function for MADDUX Chat API
// Uses user-provided Anthropic API key

const CONTEXT = `
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

VALIDATED PERFORMANCE:
- Correlation: 0.50
- R-squared: 28%
- Walk-Forward Hit Rate: 79.3%

BACKTEST RESULTS:
  2018→2019: 85% hit rate, avg OPS change +0.111
  2019→2020: 80% hit rate, avg OPS change +0.055
  2020→2021: 85% hit rate, avg OPS change +0.101
  2021→2022: 60% hit rate, avg OPS change +0.019
  2022→2023: 95% hit rate, avg OPS change +0.101
  2023→2024: 65% hit rate, avg OPS change +0.024
  2024→2025: 85% hit rate, avg OPS change +0.071

TOP 2026 BREAKOUT CANDIDATES:
  1. LaMonte Wade Jr. (Age 31): 2025 OPS 0.524 → Predicted +0.121 → 2026 OPS 0.645 | Factor: High deviation from baseline (+0.25)
  2. Joc Pederson (Age 33): 2025 OPS 0.614 → Predicted +0.114 → 2026 OPS 0.728 | Factor: Strong career peak deviation (+0.29)
  3. Henry Davis (Age 25): 2025 OPS 0.512 → Predicted +0.105 → 2026 OPS 0.617 | Factor: Large underperformance gap (67)
  4. Anthony Santander (Age 30): 2025 OPS 0.565 → Predicted +0.092 → 2026 OPS 0.657 | Factor: Deviation from baseline (+0.24)
  5. Tyler O'Neill (Age 30): 2025 OPS 0.684 → Predicted +0.091 → 2026 OPS 0.775 | Factor: Career peak deviation (+0.23)
  6. Jose Iglesias (Age 35): 2025 OPS 0.592 → Predicted +0.090 → 2026 OPS 0.682 | Factor: Underperformance vs expected stats
  7. Jordan Walker (Age 23): 2025 OPS 0.584 → Predicted +0.089 → 2026 OPS 0.673 | Factor: Young age + high ceiling
  8. Randal Grichuk (Age 33): 2025 OPS 0.674 → Predicted +0.088 → 2026 OPS 0.762 | Factor: Career peak deviation
  9. Nolan Jones (Age 27): 2025 OPS 0.600 → Predicted +0.082 → 2026 OPS 0.682 | Factor: Underperformance gap
  10. Matt McLain (Age 25): 2025 OPS 0.643 → Predicted +0.079 → 2026 OPS 0.722 | Factor: Deviation from baseline (+0.22)

KEY INSIGHTS:
- Original MADDUX formula had NEGATIVE correlation (rejected)
- Players who improved recently tend to REGRESS (improvement_momentum is negative)
- Underperformers (xwOBA > wOBA) bounce back
- Young players with high deviation from baseline have most upside
`;

const SYSTEM_PROMPT = `You are a baseball analytics expert for AutoCoach LLC, helping users understand the MADDUX™ predictive model.

Your role:
1. Answer questions about player projections, model methodology, and accuracy
2. Be specific with player names, numbers, and statistical details
3. Explain reasoning using feature importance data
4. Note that "improvement momentum" is NEGATIVE - recent improvers regress

Keep responses concise but informative (2-3 paragraphs max).`;

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Health check
  if (req.method === 'GET') {
    return res.status(200).json({ status: 'healthy', service: 'MADDUX Chat API' });
  }

  // Handle POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { question, api_key } = req.body;

    // Validate API key
    if (!api_key || !api_key.startsWith('sk-ant-')) {
      return res.status(400).json({ error: "Invalid API key format. Key should start with 'sk-ant-'" });
    }

    // Validate question
    if (!question || question.trim().length < 3) {
      return res.status(400).json({ error: 'Question is too short' });
    }

    // Call Anthropic API
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1000,
        system: SYSTEM_PROMPT,
        messages: [{
          role: 'user',
          content: `${CONTEXT}\n\nUSER QUESTION: ${question}\n\nProvide a helpful, data-driven response based on the MADDUX model context above.`
        }]
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      if (response.status === 401) {
        return res.status(401).json({ error: 'Invalid API key. Please check your Anthropic API key.' });
      }
      if (response.status === 429) {
        return res.status(429).json({ error: 'Rate limit exceeded. Please wait and try again.' });
      }
      return res.status(response.status).json({ error: errorData.error?.message || 'API request failed' });
    }

    const data = await response.json();
    const answer = data.content?.[0]?.text || 'No response generated';

    return res.status(200).json({ answer });

  } catch (error) {
    console.error('Error:', error);
    return res.status(500).json({ error: `Server error: ${error.message}` });
  }
}

