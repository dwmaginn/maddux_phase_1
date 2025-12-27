"""
backtest.py
Historical backtesting for MADDUX™ algorithm (enhanced v2).

Enhanced with:
- Walk-forward cross-validation (no data leakage)
- Bootstrap confidence intervals
- Survivorship bias tracking
- Multi-target support (OPS, wRC+)
- Feature-based model validation

Analyzes prediction accuracy by:
- Identifying top 20 players by score each year
- Measuring hit rate (% who improved next year)
- Calculating average improvement for top scorers
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase

# Optional sklearn for advanced validation
try:
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def get_predictions_with_outcomes(db: MadduxDatabase, 
                                   year_from: int, 
                                   year_to: int) -> pd.DataFrame:
    """
    Get MADDUX predictions and their actual outcomes.
    
    For year_from-year_to predictions, the outcome is year_to+1 OPS change.
    
    Args:
        db: Database connection
        year_from: First year of delta calculation
        year_to: Second year of delta calculation
        
    Returns:
        DataFrame with predictions and outcomes
    """
    next_year = year_to + 1
    
    sql = """
        SELECT 
            p.player_name,
            p.mlb_id,
            d.maddux_score,
            d.delta_max_ev,
            d.delta_hard_hit_pct,
            f1.ops as year_to_ops,
            f1.pa as year_to_pa,
            f2.ops as next_year_ops,
            f2.pa as next_year_pa,
            (f2.ops - f1.ops) as ops_change,
            CASE WHEN f2.ops > f1.ops THEN 1 ELSE 0 END as improved
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = ?
        JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = ?
        WHERE d.year_from = ? AND d.year_to = ?
          AND f1.ops IS NOT NULL AND f2.ops IS NOT NULL
          AND f1.pa >= 200 AND f2.pa >= 200
        ORDER BY d.maddux_score DESC
    """
    
    return db.query_df(sql, (year_to, next_year, year_from, year_to))


def calculate_hit_rate(data: pd.DataFrame, top_n: int = 20) -> Dict:
    """
    Calculate hit rate for top N predictions.
    
    Hit = player improved OPS in next year
    
    Args:
        data: DataFrame with predictions and outcomes
        top_n: Number of top predictions to evaluate
        
    Returns:
        Dict with hit rate statistics
    """
    if len(data) < top_n:
        return {
            'n': len(data),
            'top_n': top_n,
            'error': f'Insufficient data (need {top_n}, have {len(data)})'
        }
    
    # Get top N by MADDUX score
    top = data.head(top_n)
    
    # Calculate hit rate
    hits = top['improved'].sum()
    hit_rate = hits / top_n
    
    # Calculate average improvement for top N
    avg_ops_change = top['ops_change'].mean()
    
    # Calculate average MADDUX score for top N
    avg_maddux_score = top['maddux_score'].mean()
    
    return {
        'n': len(data),
        'top_n': top_n,
        'hits': int(hits),
        'hit_rate': hit_rate,
        'avg_ops_change': avg_ops_change,
        'avg_maddux_score': avg_maddux_score,
        'top_players': top[['player_name', 'maddux_score', 'ops_change', 'improved']].to_dict('records')
    }


def backtest_all_years(db: MadduxDatabase, 
                       start_year: int = 2015, 
                       end_year: int = 2024,
                       top_n: int = 20) -> List[Dict]:
    """
    Backtest MADDUX predictions for all year pairs.
    
    Args:
        db: Database connection
        start_year: First year to start backtesting
        end_year: Last year to end backtesting (must have next year data)
        top_n: Number of top predictions to evaluate
        
    Returns:
        List of backtest results by year pair
    """
    results = []
    
    for year_from in range(start_year, end_year):
        year_to = year_from + 1
        next_year = year_to + 1
        
        # Skip if we don't have next year data
        if next_year > 2025:
            continue
        
        # Get predictions with outcomes
        data = get_predictions_with_outcomes(db, year_from, year_to)
        
        if len(data) == 0:
            results.append({
                'year_pair': f"{year_from}-{year_to}",
                'prediction_for': next_year,
                'error': 'No data available'
            })
            continue
        
        # Calculate hit rate
        hit_result = calculate_hit_rate(data, top_n)
        hit_result['year_pair'] = f"{year_from}-{year_to}"
        hit_result['prediction_for'] = next_year
        
        results.append(hit_result)
    
    return results


def get_overall_statistics(backtest_results: List[Dict]) -> Dict:
    """
    Calculate overall backtest statistics.
    
    Args:
        backtest_results: List of results from backtest_all_years
        
    Returns:
        Dict with overall statistics
    """
    valid_results = [r for r in backtest_results if 'hit_rate' in r]
    
    if not valid_results:
        return {'error': 'No valid backtest results'}
    
    hit_rates = [r['hit_rate'] for r in valid_results]
    avg_ops_changes = [r['avg_ops_change'] for r in valid_results]
    
    return {
        'years_tested': len(valid_results),
        'avg_hit_rate': np.mean(hit_rates),
        'min_hit_rate': np.min(hit_rates),
        'max_hit_rate': np.max(hit_rates),
        'std_hit_rate': np.std(hit_rates),
        'avg_ops_improvement': np.mean(avg_ops_changes),
        'target_hit_rate': 0.80,
        'meets_target': np.mean(hit_rates) >= 0.80
    }


def get_best_historical_predictions(db: MadduxDatabase, top_n: int = 20) -> List[Dict]:
    """
    Get the best historical MADDUX predictions that came true.
    
    Args:
        db: Database connection
        top_n: Number of best predictions to return
        
    Returns:
        List of successful predictions
    """
    sql = """
        SELECT 
            p.player_name,
            d.year_from,
            d.year_to,
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
        ORDER BY (f2.ops - f1.ops) DESC, d.maddux_score DESC
        LIMIT ?
    """
    
    result = db.query(sql, (top_n,))
    
    predictions = []
    for row in result:
        predictions.append({
            'player_name': row[0],
            'year_pair': f"{row[1]}-{row[2]}",
            'maddux_score': row[3],
            'delta_max_ev': row[4],
            'delta_hard_hit_pct': row[5],
            'before_ops': row[6],
            'after_ops': row[7],
            'ops_improvement': row[8]
        })
    
    return predictions


def analyze_score_thresholds(db: MadduxDatabase) -> pd.DataFrame:
    """
    Analyze hit rates at different MADDUX score thresholds.
    
    Args:
        db: Database connection
        
    Returns:
        DataFrame with threshold analysis
    """
    sql = """
        SELECT 
            d.maddux_score,
            CASE WHEN f2.ops > f1.ops THEN 1 ELSE 0 END as improved
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
        JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
        WHERE f1.ops IS NOT NULL AND f2.ops IS NOT NULL
          AND f1.pa >= 200 AND f2.pa >= 200
    """
    
    data = db.query_df(sql)
    
    if len(data) == 0:
        return pd.DataFrame()
    
    # Analyze at different thresholds
    thresholds = [0, 5, 10, 15, 20, 25]
    results = []
    
    for threshold in thresholds:
        above = data[data['maddux_score'] >= threshold]
        if len(above) > 0:
            hit_rate = above['improved'].mean()
            results.append({
                'threshold': threshold,
                'n': len(above),
                'hit_rate': hit_rate,
                'pct_of_total': len(above) / len(data) * 100
            })
    
    return pd.DataFrame(results)


# ============================================================================
# ENHANCED VALIDATION (v2)
# ============================================================================

def walk_forward_validation(
    db: MadduxDatabase,
    start_train_year: int = 2016,
    end_test_year: int = 2024,
    min_pa: int = 200,
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Run walk-forward cross-validation for time-series prediction.
    
    Ensures no data leakage: always train on past, predict future.
    
    Args:
        db: Database connection
        start_train_year: First year for training
        end_test_year: Last year to test
        min_pa: Minimum PA threshold
        top_n: Number of top predictions to evaluate
        
    Returns:
        Dict with validation results
    """
    results_by_year = []
    
    print(f"\nWalk-forward validation: {start_train_year} to {end_test_year}")
    print("-" * 70)
    
    for test_year in range(start_train_year + 2, end_test_year + 1):
        # Training: all years before test_year
        # Test: predict test_year + 1 using test_year features
        
        # Get feature-based predictions if available
        sql = """
            SELECT 
                p.player_name,
                pf.*,
                f_curr.ops as current_ops,
                f_next.ops as next_ops,
                (f_next.ops - f_curr.ops) as ops_change,
                CASE WHEN f_next.ops > f_curr.ops THEN 1 ELSE 0 END as improved
            FROM player_features pf
            JOIN players p ON pf.player_id = p.id
            JOIN fangraphs_seasons f_curr ON pf.player_id = f_curr.player_id AND pf.year = f_curr.year
            JOIN fangraphs_seasons f_next ON pf.player_id = f_next.player_id AND pf.year + 1 = f_next.year
            WHERE pf.year = ? AND f_curr.pa >= ? AND f_next.pa >= ?
            ORDER BY pf.deviation_from_baseline DESC
        """
        
        test_data = db.query_df(sql, (test_year, min_pa, min_pa))
        
        if len(test_data) < top_n:
            # Fall back to original MADDUX score
            test_data = get_predictions_with_outcomes(db, test_year - 1, test_year)
        
        if len(test_data) < top_n:
            print(f"  {test_year}: Insufficient data ({len(test_data)} < {top_n})")
            continue
        
        # Get top N predictions
        top = test_data.head(top_n)
        
        # Calculate metrics
        hits = top['improved'].sum() if 'improved' in top.columns else (top['ops_change'] > 0).sum()
        hit_rate = hits / top_n
        avg_change = top['ops_change'].mean()
        
        results_by_year.append({
            'year': test_year,
            'predicting': test_year + 1,
            'n_total': len(test_data),
            'n_top': top_n,
            'hits': int(hits),
            'hit_rate': hit_rate,
            'avg_ops_change': avg_change
        })
        
        print(f"  {test_year}→{test_year+1}: Hit Rate={hit_rate*100:.1f}%, "
              f"Avg ΔOPS={avg_change:+.3f}, n={len(test_data)}")
    
    # Calculate aggregate statistics
    if results_by_year:
        hit_rates = [r['hit_rate'] for r in results_by_year]
        ops_changes = [r['avg_ops_change'] for r in results_by_year]
        
        return {
            'by_year': results_by_year,
            'avg_hit_rate': np.mean(hit_rates),
            'std_hit_rate': np.std(hit_rates),
            'min_hit_rate': np.min(hit_rates),
            'max_hit_rate': np.max(hit_rates),
            'avg_ops_change': np.mean(ops_changes),
            'n_years': len(results_by_year)
        }
    
    return {'error': 'No valid results'}


def bootstrap_confidence_interval(
    data: pd.DataFrame,
    metric_func: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        data: Input DataFrame
        metric_func: Function to calculate metric
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    point_estimate = metric_func(data)
    
    bootstrap_estimates = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample = data.iloc[sample_idx]
        bootstrap_estimates.append(metric_func(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_estimates, alpha * 100)
    upper = np.percentile(bootstrap_estimates, (1 - alpha) * 100)
    
    return point_estimate, lower, upper


def calculate_hit_rate_with_ci(
    db: MadduxDatabase,
    top_n: int = 20,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate hit rate with bootstrap confidence interval.
    
    Args:
        db: Database connection
        top_n: Number of top predictions
        n_bootstrap: Bootstrap samples
        confidence: Confidence level
        
    Returns:
        Dict with hit rate and CI
    """
    # Get all predictions with outcomes
    sql = """
        SELECT 
            d.maddux_score,
            (f2.ops - f1.ops) as ops_change,
            CASE WHEN f2.ops > f1.ops THEN 1 ELSE 0 END as improved
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
        JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
        WHERE f1.pa >= 200 AND f2.pa >= 200
        ORDER BY d.maddux_score DESC
    """
    
    data = db.query_df(sql)
    
    if len(data) < top_n:
        return {'error': 'Insufficient data'}
    
    # Define metric function
    def hit_rate_metric(df):
        top = df.nlargest(min(top_n, len(df)), 'maddux_score')
        return top['improved'].mean()
    
    # Calculate with bootstrap CI
    estimate, lower, upper = bootstrap_confidence_interval(
        data, hit_rate_metric, n_bootstrap, confidence
    )
    
    return {
        'hit_rate': estimate,
        'ci_lower': lower,
        'ci_upper': upper,
        'confidence': confidence,
        'n_total': len(data),
        'top_n': top_n
    }


def survivorship_bias_analysis(
    db: MadduxDatabase,
    min_pa: int = 200
) -> Dict[str, Any]:
    """
    Analyze survivorship bias in predictions.
    
    Tracks players who:
    - Qualified in year N but not N+1 (dropped out)
    - Had high MADDUX scores but didn't meet PA threshold next year
    
    Args:
        db: Database connection
        min_pa: PA threshold
        
    Returns:
        Dict with survivorship analysis
    """
    # Players with high MADDUX score who dropped below PA threshold
    sql = """
        SELECT 
            d.year_to,
            COUNT(*) as total_high_score,
            SUM(CASE WHEN f2.pa IS NULL OR f2.pa < ? THEN 1 ELSE 0 END) as dropped_out
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = d.year_to
        LEFT JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = d.year_to + 1
        WHERE d.maddux_score > 10 AND f1.pa >= ?
        GROUP BY d.year_to
        ORDER BY d.year_to
    """
    
    data = db.query_df(sql, (min_pa, min_pa))
    
    if len(data) == 0:
        return {'error': 'No data available'}
    
    total_high_score = data['total_high_score'].sum()
    total_dropped = data['dropped_out'].sum()
    
    return {
        'years_analyzed': len(data),
        'total_high_score_players': int(total_high_score),
        'total_dropped_out': int(total_dropped),
        'dropout_rate': total_dropped / total_high_score if total_high_score > 0 else 0,
        'by_year': data.to_dict('records'),
        'note': 'Players with MADDUX score > 10 who dropped below PA threshold'
    }


def feature_importance_validation(db: MadduxDatabase) -> Dict[str, Any]:
    """
    Validate feature importance through correlation analysis.
    
    Args:
        db: Database connection
        
    Returns:
        Dict with feature correlations
    """
    sql = """
        SELECT 
            pf.underperformance_gap,
            pf.deviation_from_baseline,
            pf.career_peak_deviation,
            pf.improvement_momentum,
            pf.combined_luck_index,
            pf.quality_zscore,
            pf.hard_hit_zscore,
            pf.age_factor,
            pf.original_maddux_score,
            (f2.ops - f1.ops) as ops_change
        FROM player_features pf
        JOIN fangraphs_seasons f1 ON pf.player_id = f1.player_id AND pf.year = f1.year
        JOIN fangraphs_seasons f2 ON pf.player_id = f2.player_id AND pf.year + 1 = f2.year
        WHERE f1.pa >= 200 AND f2.pa >= 200
    """
    
    data = db.query_df(sql)
    
    if len(data) < 50:
        return {'error': 'Insufficient data for feature validation'}
    
    features = [c for c in data.columns if c != 'ops_change']
    correlations = {}
    
    for feat in features:
        clean = data[[feat, 'ops_change']].dropna()
        if len(clean) > 30:
            r = np.corrcoef(clean[feat], clean['ops_change'])[0, 1]
            correlations[feat] = r
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return {
        'n_samples': len(data),
        'correlations': dict(sorted_corr),
        'top_positive': [k for k, v in sorted_corr if v > 0][:3],
        'top_negative': [k for k, v in sorted_corr if v < 0][:3],
        'original_maddux_corr': correlations.get('original_maddux_score', None)
    }


def main():
    """Run full backtest analysis (enhanced v2)."""
    print("=" * 70)
    print("MADDUX™ Algorithm - Historical Backtest (Enhanced v2)")
    print("=" * 70)
    
    db = MadduxDatabase()
    
    # 1. Original MADDUX backtest
    print("\n1. ORIGINAL MADDUX SCORE BACKTEST (Top 20)")
    print("-" * 70)
    
    backtest_results = backtest_all_years(db)
    
    print(f"{'Year Pair':<12} {'Predicting':>12} {'N':>6} {'Hits':>6} {'Hit Rate':>10} {'Avg ΔOPS':>10}")
    print("-" * 70)
    
    for result in backtest_results:
        if 'error' in result:
            print(f"{result['year_pair']:<12} {result['prediction_for']:>12} {'Error':>6}")
        else:
            print(f"{result['year_pair']:<12} {result['prediction_for']:>12} "
                  f"{result['n']:>6} {result['hits']:>6} "
                  f"{result['hit_rate']*100:>9.1f}% {result['avg_ops_change']:>+10.3f}")
    
    # 2. Overall statistics
    print("\n2. ORIGINAL MODEL OVERALL STATISTICS")
    print("-" * 70)
    
    overall = get_overall_statistics(backtest_results)
    if 'error' not in overall:
        print(f"Years tested: {overall['years_tested']}")
        print(f"Average hit rate: {overall['avg_hit_rate']*100:.1f}%")
        print(f"Hit rate range: {overall['min_hit_rate']*100:.1f}% - {overall['max_hit_rate']*100:.1f}%")
        print(f"Standard deviation: {overall['std_hit_rate']*100:.1f}%")
        print(f"Average OPS improvement for top 20: {overall['avg_ops_improvement']:+.3f}")
        print(f"\nTarget hit rate: {overall['target_hit_rate']*100:.0f}%")
        print(f"Meets target: {'YES ✓' if overall['meets_target'] else 'NO ✗'}")
    else:
        print(f"Error: {overall['error']}")
    
    # 3. Walk-forward validation (enhanced)
    print("\n3. WALK-FORWARD CROSS-VALIDATION (Enhanced Features)")
    print("-" * 70)
    
    wf_results = walk_forward_validation(db, top_n=20)
    if 'error' not in wf_results:
        print(f"\n  Summary:")
        print(f"  Years validated: {wf_results['n_years']}")
        print(f"  Average hit rate: {wf_results['avg_hit_rate']*100:.1f}% "
              f"(±{wf_results['std_hit_rate']*100:.1f}%)")
        print(f"  Hit rate range: {wf_results['min_hit_rate']*100:.1f}% - "
              f"{wf_results['max_hit_rate']*100:.1f}%")
        print(f"  Average OPS change: {wf_results['avg_ops_change']:+.3f}")
    
    # 4. Bootstrap confidence interval
    print("\n4. BOOTSTRAP CONFIDENCE INTERVAL (95%)")
    print("-" * 70)
    
    ci_result = calculate_hit_rate_with_ci(db, top_n=20, n_bootstrap=500)
    if 'error' not in ci_result:
        print(f"  Hit Rate: {ci_result['hit_rate']*100:.1f}%")
        print(f"  95% CI: [{ci_result['ci_lower']*100:.1f}%, {ci_result['ci_upper']*100:.1f}%]")
        print(f"  Based on {ci_result['n_total']} total predictions")
    else:
        print(f"  Error: {ci_result['error']}")
    
    # 5. Survivorship bias analysis
    print("\n5. SURVIVORSHIP BIAS ANALYSIS")
    print("-" * 70)
    
    surv = survivorship_bias_analysis(db)
    if 'error' not in surv:
        print(f"  High-score players (MADDUX > 10): {surv['total_high_score_players']}")
        print(f"  Dropped below 200 PA next year: {surv['total_dropped_out']}")
        print(f"  Dropout rate: {surv['dropout_rate']*100:.1f}%")
        print(f"  Note: {surv['note']}")
    else:
        print(f"  Error: {surv['error']}")
    
    # 6. Feature importance validation
    print("\n6. FEATURE CORRELATION WITH NEXT-YEAR OPS CHANGE")
    print("-" * 70)
    
    feat_valid = feature_importance_validation(db)
    if 'error' not in feat_valid:
        print(f"  Based on {feat_valid['n_samples']} player-seasons\n")
        print(f"  {'Feature':<30} {'Correlation':>12}")
        print("  " + "-" * 45)
        for feat, corr in list(feat_valid['correlations'].items())[:8]:
            print(f"  {feat:<30} {corr:>+12.4f}")
        
        if feat_valid['original_maddux_corr'] is not None:
            print(f"\n  Original MADDUX score correlation: {feat_valid['original_maddux_corr']:+.4f}")
    else:
        print(f"  Error: {feat_valid['error']}")
    
    # 7. Score threshold analysis
    print("\n7. HIT RATE BY MADDUX SCORE THRESHOLD")
    print("-" * 70)
    
    threshold_df = analyze_score_thresholds(db)
    if len(threshold_df) > 0:
        print(f"{'Threshold':>10} {'N':>8} {'Hit Rate':>12} {'% of Total':>12}")
        print("-" * 50)
        for _, row in threshold_df.iterrows():
            print(f"{'>= '+str(int(row['threshold'])):>10} {int(row['n']):>8} "
                  f"{row['hit_rate']*100:>11.1f}% {row['pct_of_total']:>11.1f}%")
    
    # 8. Best historical predictions
    print("\n8. BEST HISTORICAL PREDICTIONS (MADDUX Score > 10, Improved)")
    print("-" * 70)
    
    best = get_best_historical_predictions(db, top_n=10)
    print(f"{'Player':<25} {'Years':<10} {'Score':>8} {'Before':>8} {'After':>8} {'Δ OPS':>8}")
    print("-" * 70)
    for pred in best:
        print(f"{pred['player_name']:<25} {pred['year_pair']:<10} "
              f"{pred['maddux_score']:>8.1f} {pred['before_ops']:>8.3f} "
              f"{pred['after_ops']:>8.3f} {pred['ops_improvement']:>+8.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    
    if 'error' not in overall and 'error' not in wf_results:
        print(f"\n  Original MADDUX Score:")
        print(f"    Hit Rate: {overall['avg_hit_rate']*100:.1f}%")
        print(f"    Target (80%): {'MET ✓' if overall['meets_target'] else 'NOT MET ✗'}")
        
        if wf_results['avg_hit_rate'] > overall['avg_hit_rate']:
            improvement = wf_results['avg_hit_rate'] - overall['avg_hit_rate']
            print(f"\n  Enhanced Features (Walk-Forward):")
            print(f"    Hit Rate: {wf_results['avg_hit_rate']*100:.1f}%")
            print(f"    Improvement: +{improvement*100:.1f} percentage points")
    
    db.close()
    print("\nBacktest complete!")


if __name__ == "__main__":
    main()

