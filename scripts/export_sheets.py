"""
export_sheets.py
Generate Google Sheets formatted CSVs for all required tabs.

Output tabs:
1. Raw - Hitters (2015-2025)
2. Calculated Deltas and Scores
3. OLS Regression Results
4. Backtest Analysis
5. 2026 Projections
6. Model Experiments (findings)
7. Service Design Specs
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase

OUTPUT_DIR = Path(__file__).parent.parent / "sheets_output"


def export_raw_hitters(db: MadduxDatabase) -> None:
    """Export raw hitter data 2015-2025."""
    sql = """
        SELECT 
            p.player_name as Player,
            p.mlb_id as MLB_ID,
            s.year as Year,
            s.max_ev as Max_EV,
            s.avg_ev as Avg_EV,
            s.hard_hit_pct as Hard_Hit_Pct,
            s.barrel_pct as Barrel_Pct,
            f.pa as PA,
            f.avg as AVG,
            f.obp as OBP,
            f.slg as SLG,
            f.ops as OPS,
            f.wrc_plus as wRC_Plus
        FROM players p
        JOIN statcast_seasons s ON p.id = s.player_id
        LEFT JOIN fangraphs_seasons f ON p.id = f.player_id AND s.year = f.year
        ORDER BY s.year, p.player_name
    """
    
    df = db.query_df(sql)
    df.to_csv(OUTPUT_DIR / "1_raw_hitters.csv", index=False)
    print(f"Exported raw hitters: {len(df)} rows")


def export_deltas_scores(db: MadduxDatabase) -> None:
    """Export calculated deltas and MADDUX scores."""
    sql = """
        SELECT 
            p.player_name as Player,
            p.mlb_id as MLB_ID,
            d.year_from as Year_From,
            d.year_to as Year_To,
            ROUND(d.delta_max_ev, 2) as Delta_Max_EV,
            ROUND(d.delta_hard_hit_pct, 2) as Delta_Hard_Hit_Pct,
            ROUND(d.maddux_score, 2) as MADDUX_Score,
            ROUND(d.delta_ops, 3) as Delta_OPS
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        ORDER BY d.year_from, d.maddux_score DESC
    """
    
    df = db.query_df(sql)
    df.to_csv(OUTPUT_DIR / "2_deltas_scores.csv", index=False)
    print(f"Exported deltas/scores: {len(df)} rows")


def export_regression_results(db: MadduxDatabase) -> None:
    """Export OLS regression results."""
    # Calculate pooled regression stats
    sql = """
        SELECT 
            d.maddux_score,
            d.delta_max_ev,
            d.delta_hard_hit_pct,
            (f2.ops - f1.ops) as next_year_ops_change
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
        JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
        WHERE f1.ops IS NOT NULL AND f2.ops IS NOT NULL
    """
    
    data = db.query(sql)
    
    if data:
        scores = [r[0] for r in data]
        changes = [r[3] for r in data]
        correlation = np.corrcoef(scores, changes)[0, 1]
    else:
        correlation = 0
    
    # Create summary DataFrame
    results = pd.DataFrame([
        {"Metric": "Sample Size (n)", "Value": len(data) if data else 0},
        {"Metric": "R-squared", "Value": round(correlation**2, 4)},
        {"Metric": "Correlation (r)", "Value": round(correlation, 4)},
        {"Metric": "Proposed Ratio", "Value": 2.1},
        {"Metric": "Derived Ratio", "Value": 0.84},
        {"Metric": "Target Correlation", "Value": 0.65},
        {"Metric": "Meets Target", "Value": "NO"},
        {"Metric": "", "Value": ""},
        {"Metric": "YEARLY BREAKDOWN", "Value": ""},
    ])
    
    # Add yearly results
    yearly_sql = """
        SELECT 
            d.year_from || '-' || d.year_to as year_pair,
            COUNT(*) as n,
            ROUND(AVG(CASE WHEN f2.ops > f1.ops THEN 1.0 ELSE 0.0 END) * 100, 1) as hit_rate
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
        JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
        WHERE d.maddux_score >= 10 AND f1.ops IS NOT NULL AND f2.ops IS NOT NULL
        GROUP BY d.year_from, d.year_to
        ORDER BY d.year_from
    """
    
    yearly = db.query(yearly_sql)
    for year_pair, n, hit_rate in yearly:
        results = pd.concat([results, pd.DataFrame([{
            "Metric": f"Hit Rate {year_pair}", 
            "Value": f"{hit_rate}% (n={n})"
        }])], ignore_index=True)
    
    results.to_csv(OUTPUT_DIR / "3_regression_results.csv", index=False)
    print(f"Exported regression results")


def export_backtest(db: MadduxDatabase) -> None:
    """Export backtest analysis."""
    sql = """
        SELECT 
            d.year_from || '-' || d.year_to as Year_Pair,
            d.year_to + 1 as Prediction_For,
            COUNT(*) as Total_Players,
            SUM(CASE WHEN d.maddux_score >= 10 THEN 1 ELSE 0 END) as High_Score_Count,
            ROUND(AVG(CASE WHEN f2.ops > f1.ops THEN 1.0 ELSE 0.0 END) * 100, 1) as Overall_Hit_Rate,
            ROUND(AVG(CASE WHEN d.maddux_score >= 10 AND f2.ops > f1.ops THEN 1.0 
                       WHEN d.maddux_score >= 10 THEN 0.0 END) * 100, 1) as Top_Score_Hit_Rate,
            ROUND(AVG(f2.ops - f1.ops), 3) as Avg_OPS_Change
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
        JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
        WHERE f1.ops IS NOT NULL AND f2.ops IS NOT NULL
        GROUP BY d.year_from, d.year_to
        ORDER BY d.year_from
    """
    
    df = db.query_df(sql)
    df.to_csv(OUTPUT_DIR / "4_backtest_analysis.csv", index=False)
    print(f"Exported backtest: {len(df)} year-pairs")


def export_projections_2026(db: MadduxDatabase) -> None:
    """Export 2026 projections."""
    sql = """
        SELECT 
            ROW_NUMBER() OVER (ORDER BY d.maddux_score DESC) as Rank,
            p.player_name as Player,
            p.mlb_id as MLB_ID,
            ROUND(d.delta_max_ev, 1) as Delta_Max_EV,
            ROUND(d.delta_hard_hit_pct, 1) as Delta_Hard_Hit_Pct,
            ROUND(d.maddux_score, 1) as MADDUX_Score,
            ROUND(d.maddux_score * 0.002, 3) as Projected_OPS_Change,
            CASE 
                WHEN d.maddux_score > 15 AND d.delta_max_ev > 0 AND d.delta_hard_hit_pct > 0 THEN 'High'
                WHEN d.maddux_score >= 10 THEN 'Medium'
                WHEN d.maddux_score >= 5 THEN 'Low'
                ELSE 'Very Low'
            END as Confidence
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        WHERE d.year_from = 2024 AND d.year_to = 2025
        ORDER BY d.maddux_score DESC
        LIMIT 30
    """
    
    df = db.query_df(sql)
    df.to_csv(OUTPUT_DIR / "5_projections_2026.csv", index=False)
    print(f"Exported 2026 projections: {len(df)} players")


def export_experiments() -> None:
    """Export model experiments and findings."""
    experiments = pd.DataFrame([
        {
            "Experiment": "Baseline MADDUX Formula",
            "Hypothesis": "Score = ΔMaxEV + (2.1 × ΔHH%) predicts OPS improvement",
            "Result": "Correlation r=0.14, R²=1.9%",
            "Recommendation": "Model requires refinement"
        },
        {
            "Experiment": "Derived Coefficient Ratio",
            "Hypothesis": "2.1 weight for Hard Hit% is optimal",
            "Result": "Derived ratio = 0.84 from OLS",
            "Recommendation": "Consider lower weight or alternative formulation"
        },
        {
            "Experiment": "Score Threshold Analysis",
            "Hypothesis": "Higher scores = higher hit rates",
            "Result": "Counterintuitive: Higher scores had LOWER hit rates",
            "Recommendation": "Investigate why high-delta players underperform"
        },
        {
            "Experiment": "Top-20 Backtest",
            "Hypothesis": "Top 20 by score improve >80% of time",
            "Result": "Average hit rate: 32.8% (range: 10-55%)",
            "Recommendation": "Target not met, model refinement needed"
        },
        {
            "Experiment": "Alternative: Add Barrel%",
            "Hypothesis": "Barrel% may improve predictive power",
            "Result": "Not yet tested",
            "Recommendation": "Test in Phase 2"
        },
        {
            "Experiment": "Alternative: 2-Year Window",
            "Hypothesis": "2-year averages may reduce noise",
            "Result": "Not yet tested",
            "Recommendation": "Test in Phase 2"
        },
        {
            "Experiment": "Alternative: wRC+ instead of OPS",
            "Hypothesis": "wRC+ is park-adjusted, may show cleaner signal",
            "Result": "Not yet tested",
            "Recommendation": "Test in Phase 2"
        }
    ])
    
    experiments.to_csv(OUTPUT_DIR / "6_model_experiments.csv", index=False)
    print("Exported model experiments")


def export_service_design() -> None:
    """Export service design specifications."""
    specs = pd.DataFrame([
        {"Component": "DATABASE SCHEMA", "Specification": ""},
        {"Component": "Table: players", "Specification": "id, mlb_id, fangraphs_id, player_name"},
        {"Component": "Table: statcast_seasons", "Specification": "player_id, year, max_ev, hard_hit_pct, barrel_pct"},
        {"Component": "Table: fangraphs_seasons", "Specification": "player_id, year, pa, ops, wrc_plus"},
        {"Component": "Table: player_deltas", "Specification": "player_id, year_from, year_to, deltas, maddux_score"},
        {"Component": "", "Specification": ""},
        {"Component": "API ENDPOINTS", "Specification": ""},
        {"Component": "GET /players", "Specification": "List all players with optional year filter"},
        {"Component": "GET /players/{id}/seasons", "Specification": "Get all seasons for a player"},
        {"Component": "GET /projections/{year}", "Specification": "Get breakout projections for target year"},
        {"Component": "GET /backtest/{year_pair}", "Specification": "Get backtest results for year pair"},
        {"Component": "POST /calculate", "Specification": "Recalculate scores with custom weights"},
        {"Component": "", "Specification": ""},
        {"Component": "ROLLING CALCULATIONS", "Specification": ""},
        {"Component": "Daily Update", "Specification": "Pull new Statcast data, recalc deltas"},
        {"Component": "Weekly Update", "Specification": "Full refresh of all scores"},
        {"Component": "Alert Trigger", "Specification": "Score crosses threshold (e.g., >15)"},
        {"Component": "", "Specification": ""},
        {"Component": "ALERT THRESHOLDS", "Specification": ""},
        {"Component": "High Priority", "Specification": "MADDUX Score > 20, both metrics improving"},
        {"Component": "Medium Priority", "Specification": "MADDUX Score 15-20"},
        {"Component": "Watch List", "Specification": "MADDUX Score 10-15"},
        {"Component": "", "Specification": ""},
        {"Component": "DEPLOYMENT", "Specification": ""},
        {"Component": "Database", "Specification": "PostgreSQL (upgrade from SQLite for production)"},
        {"Component": "API", "Specification": "FastAPI or Flask REST API"},
        {"Component": "Dashboard", "Specification": "React or static HTML with Chart.js"},
        {"Component": "Hosting", "Specification": "AWS/GCP/Vercel"}
    ])
    
    specs.to_csv(OUTPUT_DIR / "7_service_design.csv", index=False)
    print("Exported service design specs")


def main():
    """Export all Google Sheets data."""
    print("=" * 60)
    print("Exporting Google Sheets Data")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    db = MadduxDatabase()
    
    # Export all tabs
    export_raw_hitters(db)
    export_deltas_scores(db)
    export_regression_results(db)
    export_backtest(db)
    export_projections_2026(db)
    export_experiments()
    export_service_design()
    
    db.close()
    
    print("\n" + "=" * 60)
    print(f"All exports saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nImport these CSVs into Google Sheets as separate tabs:")
    print("  1. Raw - Hitters (2015-2025)")
    print("  2. Deltas & Scores")
    print("  3. Regression Results")
    print("  4. Backtest Analysis")
    print("  5. 2026 Projections")
    print("  6. Model Experiments")
    print("  7. Service Design")


if __name__ == "__main__":
    main()

