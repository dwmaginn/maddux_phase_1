"""
export_dashboard_data.py
Export data for the MADDUX dashboard visualization.

Generates JSON data files for:
- 2026 projections
- Backtest results
- Score vs OPS scatter data
- Historical accuracy by year
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase
from scripts.projections import get_2026_projections
from scripts.backtest import backtest_all_years, get_overall_statistics


def export_projections(db: MadduxDatabase, output_dir: Path) -> None:
    """Export 2026 projections to JSON."""
    projections = get_2026_projections(db, top_n=30)
    
    # Simplify for JSON
    data = []
    for p in projections:
        data.append({
            'rank': p['rank'],
            'player': p['player_name'],
            'score': round(p['maddux_score'], 1),
            'deltaMaxEV': round(p['delta_max_ev'], 1),
            'deltaHardHit': round(p['delta_hard_hit_pct'], 1),
            'confidence': p['confidence_tier'],
            'projectedOPSChange': round(p.get('projected_ops_change', 0), 3)
        })
    
    with open(output_dir / 'projections.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(data)} projections")


def export_backtest(db: MadduxDatabase, output_dir: Path) -> None:
    """Export backtest results to JSON."""
    results = backtest_all_years(db)
    overall = get_overall_statistics(results)
    
    # Format for JSON
    yearly = []
    for r in results:
        if 'error' not in r:
            yearly.append({
                'yearPair': r['year_pair'],
                'predictionFor': r['prediction_for'],
                'n': r['n'],
                'hits': r['hits'],
                'hitRate': round(r['hit_rate'] * 100, 1),
                'avgOPSChange': round(r['avg_ops_change'], 3)
            })
    
    data = {
        'yearly': yearly,
        'overall': {
            'avgHitRate': round(overall.get('avg_hit_rate', 0) * 100, 1),
            'minHitRate': round(overall.get('min_hit_rate', 0) * 100, 1),
            'maxHitRate': round(overall.get('max_hit_rate', 0) * 100, 1),
            'yearsTeated': overall.get('years_tested', 0),
            'meetsTarget': bool(overall.get('meets_target', False))
        }
    }
    
    with open(output_dir / 'backtest.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported backtest results for {len(yearly)} years")


def export_scatter_data(db: MadduxDatabase, output_dir: Path) -> None:
    """Export MADDUX score vs OPS change scatter data."""
    sql = """
        SELECT 
            p.player_name,
            d.year_from,
            d.year_to,
            d.maddux_score,
            (f2.ops - f1.ops) as ops_change
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
        JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
        WHERE f1.ops IS NOT NULL AND f2.ops IS NOT NULL
        ORDER BY d.maddux_score DESC
    """
    
    result = db.query(sql)
    
    data = []
    for row in result:
        data.append({
            'player': row[0],
            'years': f"{row[1]}-{row[2]}",
            'score': round(row[3], 2),
            'opsChange': round(row[4], 3)
        })
    
    with open(output_dir / 'scatter.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(data)} scatter points")


def export_top_performers(db: MadduxDatabase, output_dir: Path) -> None:
    """Export historical top performers."""
    sql = """
        SELECT 
            p.player_name,
            d.year_from || '-' || d.year_to as years,
            d.maddux_score,
            d.delta_max_ev,
            d.delta_hard_hit_pct,
            f1.ops as before_ops,
            f2.ops as after_ops,
            (f2.ops - f1.ops) as ops_improvement
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
        JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
        WHERE f1.ops IS NOT NULL AND f2.ops IS NOT NULL
          AND f2.ops > f1.ops
          AND d.maddux_score > 10
        ORDER BY (f2.ops - f1.ops) DESC
        LIMIT 20
    """
    
    result = db.query(sql)
    
    data = []
    for row in result:
        data.append({
            'player': row[0],
            'years': row[1],
            'score': round(row[2], 1),
            'deltaMaxEV': round(row[3], 1),
            'deltaHardHit': round(row[4], 1),
            'beforeOPS': round(row[5], 3),
            'afterOPS': round(row[6], 3),
            'improvement': round(row[7], 3)
        })
    
    with open(output_dir / 'top_performers.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(data)} top performers")


def export_model_stats(db: MadduxDatabase, output_dir: Path) -> None:
    """Export model validation statistics."""
    # Get correlation data
    sql = """
        SELECT 
            d.maddux_score,
            (f2.ops - f1.ops) as ops_change
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
        JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
        WHERE f1.ops IS NOT NULL AND f2.ops IS NOT NULL
    """
    
    result = db.query(sql)
    
    if result:
        import numpy as np
        scores = [r[0] for r in result]
        changes = [r[1] for r in result]
        
        correlation = np.corrcoef(scores, changes)[0, 1]
        r_squared = correlation ** 2
    else:
        correlation = 0
        r_squared = 0
    
    data = {
        'n': len(result),
        'correlation': round(float(correlation), 4),
        'rSquared': round(float(r_squared), 4),
        'proposedRatio': 2.1,
        'derivedRatio': 0.84,  # From regression results
        'targetCorrelation': 0.65,
        'meetsTarget': bool(correlation >= 0.65)
    }
    
    with open(output_dir / 'model_stats.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported model statistics")


def main():
    """Export all dashboard data."""
    print("=" * 60)
    print("Exporting Dashboard Data")
    print("=" * 60)
    
    db = MadduxDatabase()
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "dashboard_data"
    output_dir.mkdir(exist_ok=True)
    
    # Export all data
    export_projections(db, output_dir)
    export_backtest(db, output_dir)
    export_scatter_data(db, output_dir)
    export_top_performers(db, output_dir)
    export_model_stats(db, output_dir)
    
    db.close()
    print(f"\nData exported to: {output_dir}")


if __name__ == "__main__":
    main()

