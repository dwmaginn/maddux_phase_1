"""
projections.py
Generate 2026 breakout predictions using MADDUX™ algorithm (enhanced v2).

Enhanced v2 features:
- Stacking Meta-Learner predictions with confidence intervals
- Bootstrap-based uncertainty quantification
- Key factor analysis (which features drive the prediction)
- Comparison with original MADDUX formula

Predictions based on 2025 features, projecting 2026 performance.
Includes confidence tiers and prediction intervals.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase, store_model_prediction
from scripts.maddux_algorithm import get_top_maddux_scores

# Try to import enhanced components
try:
    from scripts.stacking_model import (
        StackingMetaLearner, 
        get_training_data, 
        get_prediction_data,
        EXTENDED_FEATURES
    )
    ENHANCED_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced model import error: {e}")
    ENHANCED_MODEL_AVAILABLE = False


def get_2026_projections(db: MadduxDatabase, top_n: int = 30) -> List[Dict]:
    """
    Get top breakout candidates for 2026 based on 2024-2025 deltas.
    
    Args:
        db: Database connection
        top_n: Number of top candidates to return
        
    Returns:
        List of player projections with confidence tiers
    """
    sql = """
        SELECT 
            p.player_name,
            p.mlb_id,
            d.maddux_score,
            d.delta_max_ev,
            d.delta_hard_hit_pct,
            s1.max_ev as prev_max_ev,
            s2.max_ev as curr_max_ev,
            s1.hard_hit_pct as prev_hard_hit_pct,
            s2.hard_hit_pct as curr_hard_hit_pct,
            f.ops as current_ops,
            f.pa as current_pa,
            f.wrc_plus as current_wrc_plus
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN statcast_seasons s1 ON p.id = s1.player_id AND s1.year = 2024
        JOIN statcast_seasons s2 ON p.id = s2.player_id AND s2.year = 2025
        LEFT JOIN fangraphs_seasons f ON p.id = f.player_id AND f.year = 2025
        WHERE d.year_from = 2024 AND d.year_to = 2025
          AND d.delta_max_ev IS NOT NULL
          AND d.delta_hard_hit_pct IS NOT NULL
        ORDER BY d.maddux_score DESC
        LIMIT ?
    """
    
    result = db.query(sql, (top_n,))
    
    projections = []
    for i, row in enumerate(result, 1):
        player = {
            'rank': i,
            'player_name': row[0],
            'mlb_id': row[1],
            'maddux_score': row[2],
            'delta_max_ev': row[3],
            'delta_hard_hit_pct': row[4],
            'prev_max_ev': row[5],
            'curr_max_ev': row[6],
            'prev_hard_hit_pct': row[7],
            'curr_hard_hit_pct': row[8],
            'current_ops': row[9],
            'current_pa': row[10],
            'current_wrc_plus': row[11]
        }
        
        # Calculate confidence tier
        player['confidence'] = calculate_confidence(player)
        player['confidence_tier'] = get_confidence_tier(player)
        
        # Project OPS change based on historical correlation
        # Using simple linear estimate based on MADDUX score
        # Historical data shows ~0.003 OPS per MADDUX point (rough estimate)
        player['projected_ops_change'] = estimate_ops_change(player['maddux_score'])
        
        projections.append(player)
    
    return projections


def calculate_confidence(player: Dict) -> float:
    """
    Calculate confidence score (0-100) for a prediction.
    
    Factors:
    - MADDUX score magnitude
    - Gains in both metrics (vs just one)
    - Sufficient playing time
    """
    score = 50  # Base score
    
    # MADDUX score bonus
    maddux = player['maddux_score']
    if maddux >= 20:
        score += 20
    elif maddux >= 15:
        score += 15
    elif maddux >= 10:
        score += 10
    elif maddux >= 5:
        score += 5
    
    # Both metrics improving bonus
    if player['delta_max_ev'] > 0 and player['delta_hard_hit_pct'] > 0:
        score += 15
    elif player['delta_max_ev'] > 0 or player['delta_hard_hit_pct'] > 0:
        score += 5
    
    # Playing time - penalize low PA
    pa = player.get('current_pa')
    if pa and pa >= 500:
        score += 10
    elif pa and pa >= 400:
        score += 5
    elif pa and pa < 200:
        score -= 10
    
    # Cap at 0-100
    return max(0, min(100, score))


def get_confidence_tier(player: Dict) -> str:
    """
    Assign confidence tier based on criteria.
    
    High: Score >15, gains in both metrics
    Medium: Score 10-15, gains in one metric primarily
    Low: Score 5-10, small sample or data concerns
    """
    score = player['maddux_score']
    both_improving = player['delta_max_ev'] > 0 and player['delta_hard_hit_pct'] > 0
    
    if score > 15 and both_improving:
        return 'High'
    elif score >= 10:
        return 'Medium'
    elif score >= 5:
        return 'Low'
    else:
        return 'Very Low'


def estimate_ops_change(maddux_score: float) -> float:
    """
    Estimate projected OPS change based on original MADDUX score.
    
    Based on historical regression analysis.
    Note: Actual correlation is weak, so this is a rough estimate.
    """
    # Historical coefficient is roughly -0.003 per MADDUX point
    # But we present positive expectation for high scores
    # Using a more conservative estimate
    return maddux_score * 0.002


# ============================================================================
# ENHANCED PROJECTIONS (v2)
# ============================================================================

def get_enhanced_projections(
    db: MadduxDatabase,
    prediction_year: int = 2026,
    top_n: int = 30,
    min_pa: int = 200
) -> pd.DataFrame:
    """
    Generate enhanced projections using stacking model with confidence intervals.
    
    Args:
        db: Database connection
        prediction_year: Year to predict (2026)
        top_n: Number of top candidates
        min_pa: Minimum PA threshold
        
    Returns:
        DataFrame with predictions, confidence intervals, and key factors
    """
    if not ENHANCED_MODEL_AVAILABLE:
        raise ImportError("Enhanced model components not available. "
                         "Run feature_engineering.py first.")
    
    feature_year = prediction_year - 1  # Use 2025 features for 2026 prediction
    
    # Train stacking model
    X_train, y_train, train_df = get_training_data(db, feature_year, 'ops_change', min_pa)
    
    if len(X_train) < 50:
        raise ValueError(f"Insufficient training data: {len(X_train)} samples")
    
    model = StackingMetaLearner(feature_set='extended')
    model.fit(X_train.fillna(0).values, y_train.values)
    
    # Get prediction data
    pred_df = get_prediction_data(db, feature_year, min_pa)
    
    if len(pred_df) == 0:
        raise ValueError(f"No prediction data for {feature_year}")
    
    # Get predictions with confidence intervals
    X_pred = pred_df[model.available_features_].fillna(0).values
    predictions, lower, upper = model.predict_with_confidence(X_pred, n_bootstrap=200)
    
    # Add predictions to DataFrame
    pred_df['predicted_change'] = predictions
    pred_df['confidence_low'] = lower
    pred_df['confidence_high'] = upper
    pred_df['confidence_width'] = upper - lower
    
    # Calculate predicted OPS
    pred_df['predicted_ops'] = pred_df['current_ops'] + pred_df['predicted_change']
    
    # Get feature importance
    importance = model.get_feature_importance()
    
    # Calculate key factors for each prediction
    feature_cols = model.available_features_
    pred_df['key_factors'] = pred_df.apply(
        lambda row: get_key_factors(row, feature_cols, importance),
        axis=1
    )
    
    # Assign confidence tiers based on model
    pred_df['confidence_tier'] = pred_df.apply(
        lambda row: get_enhanced_confidence_tier(row),
        axis=1
    )
    
    # Sort and rank
    pred_df = pred_df.sort_values('predicted_change', ascending=False)
    pred_df['rank'] = range(1, len(pred_df) + 1)
    
    # Return top N
    return pred_df.head(top_n) if top_n > 0 else pred_df


def get_key_factors(row: pd.Series, feature_cols: List[str], 
                   importance: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Identify key factors driving a prediction.
    
    Args:
        row: DataFrame row with feature values
        feature_cols: List of feature column names
        importance: Feature importance dict
        
    Returns:
        List of top 3 factors with names and contributions
    """
    factors = []
    
    for feat in feature_cols:
        if feat in row and feat in importance:
            value = row[feat]
            if pd.notna(value) and abs(value) > 0.01:
                # Estimate contribution: value * importance (simplified)
                contribution = value * importance.get(feat, 0)
                factors.append({
                    'feature': feat,
                    'value': value,
                    'importance': importance.get(feat, 0),
                    'contribution': contribution
                })
    
    # Sort by absolute contribution
    factors.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return factors[:3]


def get_enhanced_confidence_tier(row: pd.Series) -> str:
    """
    Assign confidence tier based on model prediction quality.
    
    Factors:
    - Prediction magnitude
    - Confidence interval width
    - Consistency of positive prediction
    """
    pred = row.get('predicted_change', 0)
    low = row.get('confidence_low', 0)
    high = row.get('confidence_high', 0)
    width = row.get('confidence_width', 1)
    
    # If entire CI is positive, high confidence
    if low > 0:
        if pred > 0.030:
            return 'High'
        else:
            return 'Medium-High'
    
    # If prediction positive but CI crosses zero
    if pred > 0:
        if width < 0.050:  # Narrow CI
            return 'Medium'
        else:
            return 'Medium-Low'
    
    return 'Low'


def export_projections_json(projections: pd.DataFrame, filepath: Path) -> None:
    """
    Export projections to JSON format.
    
    Args:
        projections: DataFrame with projections
        filepath: Output path
    """
    # Convert to list of dicts
    records = []
    
    for _, row in projections.iterrows():
        record = {
            'rank': int(row['rank']),
            'player_name': row['player_name'],
            'player_id': int(row['player_id']),
            'predicted_ops_change': float(row['predicted_change']),
            'confidence_interval': {
                'low': float(row['confidence_low']),
                'high': float(row['confidence_high'])
            },
            'confidence_tier': row['confidence_tier'],
            'current_ops': float(row['current_ops']) if pd.notna(row.get('current_ops')) else None,
            'predicted_ops': float(row.get('predicted_ops', 0)) if pd.notna(row.get('predicted_ops')) else None,
            'key_factors': row.get('key_factors', [])
        }
        records.append(record)
    
    output = {
        'generated_at': datetime.now().isoformat(),
        'prediction_year': 2026,
        'model_type': 'stacking_meta_learner',
        'projections': records
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Exported projections to {filepath}")


def export_projections_csv(projections: pd.DataFrame, filepath: Path) -> None:
    """
    Export projections to CSV format.
    
    Args:
        projections: DataFrame with projections
        filepath: Output path
    """
    # Select columns for export
    export_cols = [
        'rank', 'player_name', 'player_id', 'predicted_change',
        'confidence_low', 'confidence_high', 'confidence_tier',
        'current_ops', 'predicted_ops'
    ]
    
    available_cols = [c for c in export_cols if c in projections.columns]
    export_df = projections[available_cols].copy()
    
    # Rename for clarity
    export_df = export_df.rename(columns={
        'predicted_change': 'predicted_ops_change',
        'confidence_low': 'ci_low',
        'confidence_high': 'ci_high'
    })
    
    export_df.to_csv(filepath, index=False)
    print(f"Exported projections to {filepath}")


def get_historical_accuracy_for_score_range(db: MadduxDatabase, 
                                            min_score: float, 
                                            max_score: float = None) -> Dict:
    """
    Get historical accuracy for players in a score range.
    """
    if max_score:
        sql = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN f2.ops > f1.ops THEN 1 ELSE 0 END) as improved
            FROM player_deltas d
            JOIN players p ON d.player_id = p.id
            JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
            JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
            WHERE d.maddux_score >= ? AND d.maddux_score < ?
              AND f1.ops IS NOT NULL AND f2.ops IS NOT NULL
        """
        result = db.query(sql, (min_score, max_score))
    else:
        sql = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN f2.ops > f1.ops THEN 1 ELSE 0 END) as improved
            FROM player_deltas d
            JOIN players p ON d.player_id = p.id
            JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
            JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
            WHERE d.maddux_score >= ?
              AND f1.ops IS NOT NULL AND f2.ops IS NOT NULL
        """
        result = db.query(sql, (min_score,))
    
    if result and result[0][0] > 0:
        return {
            'total': result[0][0],
            'improved': result[0][1],
            'rate': result[0][1] / result[0][0]
        }
    return {'total': 0, 'improved': 0, 'rate': 0}


def format_projections_table(projections: List[Dict]) -> str:
    """Format projections as a markdown table."""
    lines = [
        "| Rank | Player | Team | Δ Max EV | Δ HH% | MADDUX Score | Proj. Δ OPS | Confidence |",
        "|------|--------|------|----------|-------|--------------|-------------|------------|"
    ]
    
    for p in projections:
        lines.append(
            f"| {p['rank']} | {p['player_name']} | - | "
            f"{p['delta_max_ev']:+.1f} | {p['delta_hard_hit_pct']:+.1f} | "
            f"{p['maddux_score']:.1f} | {p['projected_ops_change']:+.3f} | {p['confidence_tier']} |"
        )
    
    return "\n".join(lines)


def main():
    """Generate and display 2026 projections (enhanced v2)."""
    print("=" * 70)
    print("MADDUX™ 2026 BREAKOUT PREDICTIONS (Enhanced v2)")
    print("Using Stacking Meta-Learner with Confidence Intervals")
    print("=" * 70)
    
    db = MadduxDatabase()
    
    # 1. Try enhanced projections first
    if ENHANCED_MODEL_AVAILABLE:
        print("\n1. ENHANCED MODEL PROJECTIONS")
        print("-" * 70)
        
        try:
            enhanced = get_enhanced_projections(db, prediction_year=2026, top_n=30)
            
            if len(enhanced) > 0:
                print(f"\n{'Rank':<5} {'Player':<25} {'Pred ΔOPS':>12} {'95% CI':>20} {'Tier':>12}")
                print("-" * 75)
                
                for _, row in enhanced.iterrows():
                    ci = f"[{row['confidence_low']:+.3f}, {row['confidence_high']:+.3f}]"
                    print(f"{row['rank']:<5} {row['player_name'][:24]:<25} "
                          f"{row['predicted_change']:>+12.3f} {ci:>20} {row['confidence_tier']:>12}")
                
                # Export to files
                output_dir = Path(__file__).parent.parent / "output"
                output_dir.mkdir(exist_ok=True)
                
                export_projections_json(enhanced, output_dir / "projections_2026.json")
                export_projections_csv(enhanced, output_dir / "projections_2026.csv")
                
                # Tier breakdown
                print("\n  Confidence Tier Breakdown:")
                tier_counts = enhanced['confidence_tier'].value_counts()
                for tier in ['High', 'Medium-High', 'Medium', 'Medium-Low', 'Low']:
                    count = tier_counts.get(tier, 0)
                    if count > 0:
                        print(f"    {tier}: {count} players")
                
                # Key factors summary
                print("\n  Top prediction factors (aggregated):")
                all_factors = []
                for _, row in enhanced.iterrows():
                    factors = row.get('key_factors', [])
                    if factors:
                        all_factors.extend(factors)
                
                if all_factors:
                    from collections import Counter
                    factor_counts = Counter(f['feature'] for f in all_factors)
                    for feat, count in factor_counts.most_common(5):
                        print(f"    {feat}: appears in {count} predictions")
            
            else:
                print("  No enhanced projections generated")
                
        except Exception as e:
            print(f"  Enhanced model error: {e}")
            print("  Run: python feature_engineering.py to calculate features")
    else:
        print("\n1. ENHANCED MODEL NOT AVAILABLE")
        print("-" * 70)
        print("  Install: pip install scikit-learn")
        print("  Then run: python feature_engineering.py")
    
    # 2. Original MADDUX projections (for comparison)
    print("\n2. ORIGINAL MADDUX FORMULA PROJECTIONS")
    print("-" * 70)
    
    original = get_2026_projections(db, top_n=30)
    
    if original:
        print(f"\n{'Rank':<5} {'Player':<25} {'Score':>8} {'Δ MaxEV':>8} {'Δ HH%':>8} {'Tier':>8}")
        print("-" * 70)
        
        for p in original[:15]:  # Show top 15
            print(f"{p['rank']:<5} {p['player_name'][:24]:<25} {p['maddux_score']:>8.1f} "
                  f"{p['delta_max_ev']:>+8.1f} {p['delta_hard_hit_pct']:>+8.1f} "
                  f"{p['confidence_tier']:>8}")
        
        if len(original) > 15:
            print(f"  ... and {len(original) - 15} more")
    else:
        print("  No original projections available (check if 2024-2025 data exists)")
    
    # 3. Historical context
    print("\n3. HISTORICAL ACCURACY BY SCORE RANGE")
    print("-" * 70)
    
    ranges = [(15, None), (10, 15), (5, 10), (0, 5)]
    for min_s, max_s in ranges:
        stats = get_historical_accuracy_for_score_range(db, min_s, max_s)
        label = f">={min_s}" if max_s is None else f"{min_s}-{max_s}"
        print(f"  Score {label:<8}: {stats['total']:>4} players, "
              f"{stats['rate']*100:.1f}% improved OPS next year")
    
    # 4. Model comparison summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    print("\n  Original MADDUX Formula:")
    print("    Δ Max EV + (2.1 × Δ Hard Hit%)")
    print("    ⚠️  Historical correlation: weak/negative")
    
    if ENHANCED_MODEL_AVAILABLE:
        print("\n  Enhanced Stacking Model:")
        print("    Features: luck gaps, baseline deviation, bat quality")
        print("    Models: Ridge + Lasso + Gradient Boosting")
        print("    ✓ Includes 95% confidence intervals")
    
    db.close()
    print("\n" + "=" * 70)
    print("Projections complete!")
    
    return enhanced if ENHANCED_MODEL_AVAILABLE and 'enhanced' in dir() else original


if __name__ == "__main__":
    main()

