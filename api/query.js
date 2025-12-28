// Vercel Edge Function for MADDUX Chat API
export const config = {
  runtime: 'edge',
};

const CONTEXT = `MADDUX™ ANALYTICS - AutoCoach LLC Phase 1

MODEL: Stacking Meta-Learner | Correlation 0.50 | R² 28% | Hit Rate 79.3%

KEY FEATURES:
- deviation_from_baseline (r=0.49): Current OPS vs career expected
- improvement_momentum (r=-0.46): Recent improvers tend to REGRESS
- career_peak_deviation (r=0.38): Distance from career best
- underperformance_gap (r=0.35): xwOBA minus wOBA (luck indicator)

TOP 2026 BREAKOUT CANDIDATES:
1. LaMonte Wade Jr. (+0.121 OPS)
2. Joc Pederson (+0.114 OPS)
3. Henry Davis (+0.105 OPS)
4. Anthony Santander (+0.092 OPS)
5. Tyler O'Neill (+0.091 OPS)

INSIGHT: Original MADDUX formula had negative correlation (rejected). Underperformers (xwOBA > wOBA) bounce back.`;

export default async function handler(request) {
  // Handle CORS preflight
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
      },
    });
  }

  // Health check
  if (request.method === 'GET') {
    return Response.json({ status: 'healthy', service: 'MADDUX Chat API' });
  }

  // Handle POST
  if (request.method !== 'POST') {
    return Response.json({ error: 'Method not allowed' }, { status: 405 });
  }

  try {
    const body = await request.json();
    const { question, api_key } = body;

    // Validate
    if (!api_key || !api_key.startsWith('sk-ant-')) {
      return Response.json({ error: "Invalid API key. Must start with 'sk-ant-'" }, { status: 400 });
    }
    if (!question || question.trim().length < 3) {
      return Response.json({ error: 'Question too short' }, { status: 400 });
    }

    // Call Anthropic API
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-5-20250929',
        max_tokens: 1000,
        messages: [{
          role: 'user',
          content: `You are a baseball analytics expert. Use this context:\n\n${CONTEXT}\n\nQuestion: ${question.trim()}\n\nAnswer concisely (2-3 paragraphs).`,
        }],
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      return Response.json(
        { error: data.error?.message || `API error: ${response.status}` },
        { status: response.status }
      );
    }

    return Response.json(
      { answer: data.content?.[0]?.text || 'No response' },
      {
        headers: {
          'Access-Control-Allow-Origin': '*',
        },
      }
    );
  } catch (error) {
    return Response.json({ error: error.message }, { status: 500 });
  }
}
