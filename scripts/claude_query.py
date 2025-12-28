"""
claude_query.py
Claude API integration for natural language MADDUX™ database queries.

Supports queries like:
- "Who are the top breakout candidates for 2026?"
- "Show me players with high underperformance gaps"
- "What was the prediction accuracy for 2024?"

Model: Stacking Meta-Learner (Ridge + Lasso + Gradient Boosting)
Performance: Correlation 0.50 | R² 28% | Walk-Forward Hit Rate 79.3%
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase

# Try to import anthropic
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Model performance metrics (validated)
MODEL_METRICS = {
    "correlation": 0.50,
    "r_squared": 0.28,
    "hit_rate": 0.793,
    "hit_rate_pct": "79.3%"
}

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
    {"rank": 11, "player": "Michael Conforto", "age": 32, "ops_2025": 0.637, "predicted_delta": 0.076, "projected_ops": 0.713, "key_factor": "Career peak deviation"},
    {"rank": 12, "player": "Luke Raley", "age": 30, "ops_2025": 0.631, "predicted_delta": 0.076, "projected_ops": 0.707, "key_factor": "Expected stats gap"},
    {"rank": 13, "player": "Alex Verdugo", "age": 29, "ops_2025": 0.585, "predicted_delta": 0.071, "projected_ops": 0.656, "key_factor": "Baseline deviation"},
    {"rank": 14, "player": "Marcell Ozuna", "age": 34, "ops_2025": 0.756, "predicted_delta": 0.070, "projected_ops": 0.826, "key_factor": "Continued upside"},
    {"rank": 15, "player": "Josh Rojas", "age": 31, "ops_2025": 0.512, "predicted_delta": 0.070, "projected_ops": 0.582, "key_factor": "Underperformance gap"},
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


def get_database_stats(db: MadduxDatabase) -> Dict[str, Any]:
    """
    Get database statistics for the dashboard.
    Returns counts, model metrics, and top projections.
    """
    try:
        # Get counts
        players = db.query("SELECT COUNT(DISTINCT id) FROM players")[0][0]
        statcast = db.query("SELECT COUNT(*) FROM statcast_seasons")[0][0]
        fangraphs = db.query("SELECT COUNT(*) FROM fangraphs_seasons")[0][0]
        
        # Get year range
        years = db.query("SELECT MIN(year), MAX(year) FROM statcast_seasons")
        min_year, max_year = years[0] if years else (2015, 2025)
        
        return {
            "total_players": players,
            "total_seasons": statcast,
            "fangraphs_seasons": fangraphs,
            "years_covered": f"{min_year}-{max_year}",
            "model_correlation": MODEL_METRICS["correlation"],
            "model_hit_rate": MODEL_METRICS["hit_rate"],
            "model_r_squared": MODEL_METRICS["r_squared"],
            "top_projections": TOP_2026_PROJECTIONS[:10]
        }
    except Exception as e:
        return {
            "error": str(e),
            "total_players": 0,
            "total_seasons": 0,
            "model_correlation": MODEL_METRICS["correlation"],
            "model_hit_rate": MODEL_METRICS["hit_rate"],
            "model_r_squared": MODEL_METRICS["r_squared"],
            "top_projections": TOP_2026_PROJECTIONS[:10]
        }


def get_rich_context(db: MadduxDatabase) -> str:
    """Generate comprehensive context about the database and model for Claude."""
    
    # Get database stats
    stats = get_database_stats(db)
    
    # Format top projections
    projections_text = "\n".join([
        f"  {p['rank']}. {p['player']} (Age {p['age']}): "
        f"2025 OPS {p['ops_2025']:.3f} → Predicted +{p['predicted_delta']:.3f} → "
        f"2026 OPS {p['projected_ops']:.3f} | Factor: {p['key_factor']}"
        for p in TOP_2026_PROJECTIONS[:15]
    ])
    
    # Format backtest results
    backtest_text = "\n".join([
        f"  {b['year_pair']}: {b['hit_rate']:.0f}% hit rate, avg OPS change +{b['avg_ops_change']:.3f}"
        for b in BACKTEST_RESULTS
    ])
    
    context = f"""
MADDUX™ ANALYTICS SYSTEM
========================
AutoCoach LLC | Phase 1 Validated Model

DATABASE OVERVIEW:
- Total Players: {stats['total_players']}
- Statcast Seasons: {stats['total_seasons']}
- FanGraphs Seasons: {stats.get('fangraphs_seasons', 'N/A')}
- Years Covered: {stats['years_covered']}
- Data Sources: Baseball Savant (Statcast), FanGraphs

MODEL ARCHITECTURE:
==================
The MADDUX™ model uses a Stacking Meta-Learner ensemble:
- Base Models: Ridge Regression, Lasso Regression, Gradient Boosting
- Meta-Model: Ridge Regression on out-of-fold predictions
- Features: 25+ engineered features across 3 categories

KEY FEATURES (by predictive importance):
1. deviation_from_baseline (r=0.49): Current OPS vs career expected baseline
2. improvement_momentum (r=-0.46): NEGATIVE predictor - recent improvers regress
3. career_peak_deviation (r=0.38): Distance from personal career best
4. underperformance_gap (r=0.35): xwOBA minus actual wOBA (luck adjustment)
5. combined_luck_index (r=0.33): Combined BABIP and ISO luck indicators
6. delta_launch_angle: Year-over-year launch angle changes
7. delta_z_swing: Changes in zone swing rate (discipline improvement)

VALIDATED PERFORMANCE:
=====================
- Correlation: 0.50 (predicts direction of OPS change)
- R-squared: 28% of variance explained
- Walk-Forward Hit Rate: 79.3% (near 80% target)
- Approach: Time-series cross-validation (no future data leakage)

WALK-FORWARD BACKTEST RESULTS:
{backtest_text}
Average: 79.3% hit rate across all year pairs

TOP 2026 BREAKOUT CANDIDATES:
=============================
{projections_text}

KEY INSIGHTS:
- The ORIGINAL MADDUX formula (Δ Max EV + 2.1 × Δ Hard Hit%) had NEGATIVE correlation
- Players who improved recently tend to REGRESS (improvement_momentum is negative)
- Underperforming players (xwOBA > wOBA) tend to bounce back
- Young players with high deviation from baseline have most upside
- The model identifies "unlucky" players who should improve

AVAILABLE DATA FIELDS:
- Statcast: max_ev, avg_ev, hard_hit_pct, barrel_pct, avg_launch_angle, sweet_spot_pct, 
  xwoba, xba, xslg, avg_bat_speed, swing_length, squared_up_rate, whiff_rate
- FanGraphs: pa, avg, obp, slg, ops, iso, wrc_plus, bb_pct, k_pct, contact_pct, 
  chase_rate, z_swing_pct, z_contact_pct, woba, babip, war, age
- Engineered: deviation_from_baseline, career_peak_deviation, underperformance_gap,
  improvement_momentum, combined_luck_index, bat_speed_zscore, quality_zscore
"""
    return context


def execute_query_with_key(db: MadduxDatabase, question: str, api_key: str) -> Dict[str, Any]:
    """
    Execute a natural language query using a user-provided API key.
    
    Args:
        db: Database connection
        question: User's question
        api_key: User's Anthropic API key
        
    Returns:
        Dict with 'answer' and optional 'data' fields
    """
    if not ANTHROPIC_AVAILABLE:
        return {
            "answer": "",
            "error": "The anthropic library is not installed. Please run: pip install anthropic"
        }
    
    # Get rich context
    context = get_rich_context(db)
    
    # Build the prompt
    system_prompt = """You are a baseball analytics expert for AutoCoach LLC, helping users understand the MADDUX™ predictive model for hitter breakout predictions.

Your role:
1. Answer questions about player projections, model methodology, and historical accuracy
2. Be specific with player names, numbers, and statistical details
3. Explain the reasoning behind predictions using the feature importance data
4. Clarify that the model uses walk-forward validation (no future data leakage)
5. Note that "improvement momentum" is a NEGATIVE predictor - recent improvers tend to regress

When discussing projections:
- Reference specific predicted OPS changes
- Explain the key factors driving each prediction
- Note confidence levels when relevant
- Emphasize the 79.3% walk-forward hit rate

Keep responses concise but informative. Use data from the context to support your answers."""

    user_prompt = f"""{context}

USER QUESTION: {question}

Please provide a helpful, data-driven response based on the MADDUX model and database context above."""

    try:
        # Create client with user's API key
        client = Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        answer = message.content[0].text
        
        # Check if question is about projections and include data
        data = None
        question_lower = question.lower()
        if any(word in question_lower for word in ["top", "candidates", "projections", "2026", "breakout"]):
            data = {"projections": TOP_2026_PROJECTIONS[:10]}
        elif any(word in question_lower for word in ["accuracy", "backtest", "validation", "performance"]):
            data = {"backtest": BACKTEST_RESULTS}
        
        return {
            "answer": answer,
            "data": data
        }
        
    except Exception as e:
        error_msg = str(e)
        return {
            "answer": "",
            "error": error_msg
        }


# Legacy functions for CLI usage
def get_database_context(db: MadduxDatabase) -> str:
    """Generate context about the database for Claude (legacy)."""
    return get_rich_context(db)


def execute_natural_query(db: MadduxDatabase, question: str) -> str:
    """
    Execute a natural language query against the database.
    Uses Claude to interpret the question and generate insights.
    Uses environment variable for API key (legacy CLI mode).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY environment variable not set."
    
    result = execute_query_with_key(db, question, api_key)
    
    if result.get("error"):
        return f"Error: {result['error']}"
    
    return result.get("answer", "No response generated.")


# Query templates for common questions
QUERY_TEMPLATES = {
    'top_2026': """
        SELECT p.player_name, d.maddux_score, d.delta_max_ev, d.delta_hard_hit_pct
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        WHERE d.year_from = 2024 AND d.year_to = 2025
        ORDER BY d.maddux_score DESC
        LIMIT 30
    """,
    'yearly_accuracy': """
        SELECT 
            d.year_to as prediction_year,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN f2.ops > f1.ops THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(100.0 * SUM(CASE WHEN f2.ops > f1.ops THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy_pct
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
        JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
        WHERE d.maddux_score >= 10
        GROUP BY d.year_to
        ORDER BY d.year_to
    """,
    'player_history': """
        SELECT 
            s.year,
            s.max_ev,
            s.hard_hit_pct,
            f.ops,
            f.wrc_plus
        FROM players p
        JOIN statcast_seasons s ON p.id = s.player_id
        LEFT JOIN fangraphs_seasons f ON p.id = f.player_id AND s.year = f.year
        WHERE LOWER(p.player_name) LIKE LOWER(?)
        ORDER BY s.year
    """,
    'best_predictions': """
        SELECT 
            p.player_name,
            d.year_to as prediction_year,
            d.maddux_score,
            f1.ops as before_ops,
            f2.ops as after_ops,
            ROUND(f2.ops - f1.ops, 3) as ops_improvement
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
        JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
        WHERE d.maddux_score >= 10 AND f2.ops > f1.ops
        ORDER BY (f2.ops - f1.ops) DESC
        LIMIT 20
    """,
    'underperformers': """
        SELECT 
            p.player_name,
            f.year,
            f.ops,
            s.xwoba,
            f.woba,
            ROUND(s.xwoba - f.woba, 3) as underperformance_gap
        FROM players p
        JOIN fangraphs_seasons f ON p.id = f.player_id
        JOIN statcast_seasons s ON p.id = s.player_id AND f.year = s.year
        WHERE f.year = 2025 AND s.xwoba IS NOT NULL AND f.woba IS NOT NULL AND s.xwoba > f.woba
        ORDER BY (s.xwoba - f.woba) DESC
        LIMIT 20
    """
}


def run_template_query(db: MadduxDatabase, template_name: str, 
                       params: tuple = None) -> pd.DataFrame:
    """
    Run a predefined query template.
    
    Available templates:
    - top_2026: Top breakout candidates for 2026
    - yearly_accuracy: Prediction accuracy by year
    - player_history: Career history for a player (pass name as param)
    - best_predictions: Best historical predictions that came true
    - underperformers: Players with highest xwOBA-wOBA gap (luck-based underperformance)
    """
    if template_name not in QUERY_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. "
                        f"Available: {list(QUERY_TEMPLATES.keys())}")
    
    sql = QUERY_TEMPLATES[template_name]
    
    if params:
        return db.query_df(sql, params)
    return db.query_df(sql)


def interactive_mode(db: MadduxDatabase):
    """Run interactive query mode."""
    print("\n" + "=" * 60)
    print("MADDUX™ Natural Language Query Interface")
    print("Powered by AutoCoach LLC")
    print("=" * 60)
    print("\nModel Performance:")
    print("  Correlation: 0.50 | R²: 28% | Walk-Forward Hit Rate: 79.3%")
    print("\nType your questions about the MADDUX model or type 'quit' to exit.")
    print("Example questions:")
    print("  - Who are the top breakout candidates for 2026?")
    print("  - Why is LaMonte Wade Jr. projected to improve?")
    print("  - Which young players have the highest upside?")
    print("  - What is the underperformance gap and why does it matter?")
    print()
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\nAnalyzing with Claude...")
        response = execute_natural_query(db, question)
        print("\n" + "-" * 60)
        print(response)
        print("-" * 60)


def main():
    """Demo the query interface."""
    print("=" * 60)
    print("MADDUX™ Claude API Integration")
    print("AutoCoach LLC | Phase 1 Validation")
    print("=" * 60)
    print("\nModel: Stacking Meta-Learner")
    print("Correlation: 0.50 | R²: 28% | Walk-Forward Hit Rate: 79.3%")
    
    db = MadduxDatabase()
    
    # Show database stats
    print("\n1. DATABASE STATISTICS")
    print("-" * 60)
    stats = get_database_stats(db)
    print(f"Players: {stats['total_players']}")
    print(f"Seasons: {stats['total_seasons']}")
    print(f"Years: {stats['years_covered']}")
    
    # Show top projections
    print("\n2. TOP 2026 BREAKOUT CANDIDATES")
    print("-" * 60)
    for p in TOP_2026_PROJECTIONS[:10]:
        print(f"  {p['rank']}. {p['player']}: +{p['predicted_delta']:.3f} OPS ({p['key_factor']})")
    
    # Show backtest results
    print("\n3. WALK-FORWARD VALIDATION RESULTS")
    print("-" * 60)
    for b in BACKTEST_RESULTS:
        print(f"  {b['year_pair']}: {b['hit_rate']:.0f}% hit rate")
    
    # Try Claude query if API key available
    if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
        print("\n4. CLAUDE ANALYSIS")
        print("-" * 60)
        response = execute_natural_query(
            db, 
            "What patterns do you see in the top 2026 breakout candidates? "
            "Focus on the key features driving the predictions."
        )
        print(response)
    else:
        print("\n4. CLAUDE ANALYSIS")
        print("-" * 60)
        print("Set ANTHROPIC_API_KEY environment variable to enable Claude analysis")
        print("Example: export ANTHROPIC_API_KEY='your-api-key'")
    
    db.close()


if __name__ == "__main__":
    main()
