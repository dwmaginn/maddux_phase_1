// Vercel Serverless Function for MADDUX Chat API
// Uses user-provided Anthropic API key

const CONTEXT = `
MADDUX™ ANALYTICS SYSTEM - AutoCoach LLC | Phase 1 Validated Model

MODEL: Stacking Meta-Learner (Ridge, Lasso, Gradient Boosting) with 25+ engineered features

KEY FEATURES:
1. deviation_from_baseline (r=0.49): Current OPS vs career expected
2. improvement_momentum (r=-0.46): NEGATIVE - recent improvers regress
3. career_peak_deviation (r=0.38): Distance from career best
4. underperformance_gap (r=0.35): xwOBA minus wOBA (luck)

PERFORMANCE: Correlation 0.50 | R-squared 28% | Walk-Forward Hit Rate 79.3%

TOP 2026 BREAKOUT CANDIDATES:
1. LaMonte Wade Jr. - Predicted +0.121 OPS (deviation from baseline +0.25)
2. Joc Pederson - Predicted +0.114 OPS (career peak deviation +0.29)
3. Henry Davis - Predicted +0.105 OPS (underperformance gap 67)
4. Anthony Santander - Predicted +0.092 OPS
5. Tyler O'Neill - Predicted +0.091 OPS
6. Jordan Walker - Predicted +0.089 OPS (young age + high ceiling)
7. Matt McLain - Predicted +0.079 OPS

INSIGHTS: Original MADDUX formula had NEGATIVE correlation (rejected). Players who improved recently tend to REGRESS. Underperformers (xwOBA > wOBA) bounce back.
`;

module.exports = async function handler(req, res) {
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
    const { question, api_key } = req.body || {};

    // Validate
    if (!api_key || !api_key.startsWith('sk-ant-')) {
      return res.status(400).json({ error: "Invalid API key. Must start with 'sk-ant-'" });
    }
    if (!question || question.trim().length < 3) {
      return res.status(400).json({ error: 'Question too short' });
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
        model: 'claude-sonnet-4-5-20250929',
        max_tokens: 1000,
        messages: [{
          role: 'user',
          content: `You are a baseball analytics expert. Use this context:\n\n${CONTEXT}\n\nQuestion: ${question.trim()}\n\nAnswer concisely (2-3 paragraphs).`
        }]
      })
    });

    const data = await response.json();
    
    if (!response.ok) {
      return res.status(response.status).json({ 
        error: data.error?.message || `API error: ${response.status}` 
      });
    }

    return res.status(200).json({ 
      answer: data.content?.[0]?.text || 'No response' 
    });

  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
};
