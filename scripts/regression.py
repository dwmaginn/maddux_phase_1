"""
regression.py
OLS regression validation for MADDUX™ algorithm (enhanced v2).

Multi-target support:
- OPS change (original target)
- wRC+ change (park-adjusted)
- Binary classification (improved/declined)

Validates model performance using:
- OLS regression with coefficients
- Correlation analysis
- R-squared and MAE metrics
- Classification metrics for binary target
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase

# Try statsmodels for OLS
try:
    import statsmodels.api as sm
    from scipy import stats
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")

# Try sklearn for classification
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Available target variables
TARGET_VARIABLES = {
    'ops_change': {
        'name': 'OPS Change',
        'type': 'regression',
        'description': 'Next year OPS minus current year OPS'
    },
    'wrc_plus_change': {
        'name': 'wRC+ Change',
        'type': 'regression',
        'description': 'Next year wRC+ minus current year wRC+ (park-adjusted)'
    },
    'improved_binary': {
        'name': 'Improved (Binary)',
        'type': 'classification',
        'description': '1 if next year OPS > current year OPS, else 0'
    }
}


def get_regression_data(db: MadduxDatabase, 
                        year_from: int = None, 
                        year_to: int = None,
                        target: str = 'ops_change') -> pd.DataFrame:
    """
    Get data for regression analysis with specified target variable.
    
    For MADDUX validation, we want to predict next-year performance change
    based on current year's metric improvements.
    
    Args:
        db: Database connection
        year_from: Start year for filtering (optional)
        year_to: End year for filtering (optional)
        target: Target variable ('ops_change', 'wrc_plus_change', 'improved_binary')
        
    Returns:
        DataFrame with features and target variable
    """
    # Build SQL query with all target variables
    sql = """
        SELECT 
            d.player_id,
            p.player_name,
            d.year_from,
            d.year_to,
            d.delta_max_ev,
            d.delta_hard_hit_pct,
            d.maddux_score,
            f_curr.ops as current_ops,
            f_next.ops as next_year_ops,
            (f_next.ops - f_curr.ops) as ops_change,
            f_curr.wrc_plus as current_wrc_plus,
            f_next.wrc_plus as next_year_wrc_plus,
            (f_next.wrc_plus - f_curr.wrc_plus) as wrc_plus_change,
            CASE WHEN f_next.ops > f_curr.ops THEN 1 ELSE 0 END as improved_binary,
            f_curr.pa as current_pa,
            f_next.pa as next_pa
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        JOIN fangraphs_seasons f_curr ON p.id = f_curr.player_id AND f_curr.year = d.year_to
        JOIN fangraphs_seasons f_next ON p.id = f_next.player_id AND f_next.year = d.year_to + 1
        WHERE d.delta_max_ev IS NOT NULL 
          AND d.delta_hard_hit_pct IS NOT NULL
          AND f_curr.ops IS NOT NULL
          AND f_next.ops IS NOT NULL
          AND f_curr.pa >= 200
          AND f_next.pa >= 200
    """
    
    params = []
    if year_from is not None:
        sql += " AND d.year_from >= ?"
        params.append(year_from)
    if year_to is not None:
        sql += " AND d.year_to <= ?"
        params.append(year_to)
    
    sql += " ORDER BY d.year_from, d.maddux_score DESC"
    
    if params:
        return db.query_df(sql, tuple(params))
    return db.query_df(sql)


def get_feature_regression_data(db: MadduxDatabase, 
                                year_from: int = None,
                                year_to: int = None,
                                min_pa: int = 200) -> pd.DataFrame:
    """
    Get data for regression using engineered features.
    
    Args:
        db: Database connection
        year_from: Start year for filtering
        year_to: End year for filtering
        min_pa: Minimum PA filter
        
    Returns:
        DataFrame with features and all target variables
    """
    sql = """
        SELECT 
            p.player_name,
            pf.*,
            f_curr.ops as current_ops,
            f_next.ops as next_year_ops,
            (f_next.ops - f_curr.ops) as ops_change,
            f_curr.wrc_plus as current_wrc_plus,
            f_next.wrc_plus as next_year_wrc_plus,
            (f_next.wrc_plus - f_curr.wrc_plus) as wrc_plus_change,
            CASE WHEN f_next.ops > f_curr.ops THEN 1 ELSE 0 END as improved_binary
        FROM player_features pf
        JOIN players p ON pf.player_id = p.id
        JOIN fangraphs_seasons f_curr ON pf.player_id = f_curr.player_id AND pf.year = f_curr.year
        JOIN fangraphs_seasons f_next ON pf.player_id = f_next.player_id AND pf.year + 1 = f_next.year
        WHERE f_curr.pa >= ? AND f_next.pa >= ?
    """
    
    params = [min_pa, min_pa]
    if year_from is not None:
        sql += " AND pf.year >= ?"
        params.append(year_from)
    if year_to is not None:
        sql += " AND pf.year <= ?"
        params.append(year_to)
    
    sql += " ORDER BY pf.year"
    
    return db.query_df(sql, tuple(params))


def run_ols_regression(data: pd.DataFrame) -> Dict:
    """
    Run OLS regression to derive coefficient ratios.
    
    Model: next_year_ops_change = β₀ + β₁ * delta_max_ev + β₂ * delta_hard_hit_pct
    
    Derived ratio = β₂ / β₁
    
    Args:
        data: DataFrame with regression variables
        
    Returns:
        Dict with regression results
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for OLS regression")
    
    if len(data) < 10:
        return {'error': 'Insufficient data for regression', 'n': len(data)}
    
    # Prepare data
    X = data[['delta_max_ev', 'delta_hard_hit_pct']].values
    y = data['next_year_ops_change'].values
    
    # Remove any NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    if len(X) < 10:
        return {'error': 'Insufficient valid data', 'n': len(X)}
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    # Extract coefficients
    intercept = results.params[0]
    beta_max_ev = results.params[1]
    beta_hard_hit = results.params[2]
    
    # Calculate derived ratio
    if abs(beta_max_ev) > 0.0001:
        derived_ratio = beta_hard_hit / beta_max_ev
    else:
        derived_ratio = None
    
    # Calculate additional statistics
    y_pred = results.predict(X_with_const)
    correlation = np.corrcoef(y, y_pred)[0, 1]
    mae = np.mean(np.abs(y - y_pred))
    
    # Calculate p-values
    p_max_ev = results.pvalues[1]
    p_hard_hit = results.pvalues[2]
    
    return {
        'n': len(X),
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'correlation': correlation,
        'mae': mae,
        'intercept': intercept,
        'beta_max_ev': beta_max_ev,
        'beta_hard_hit': beta_hard_hit,
        'derived_ratio': derived_ratio,
        'p_max_ev': p_max_ev,
        'p_hard_hit': p_hard_hit,
        'f_statistic': results.fvalue,
        'f_pvalue': results.f_pvalue,
        'proposed_ratio': 2.1,
        'model_summary': str(results.summary())
    }


def run_yearly_regressions(db: MadduxDatabase) -> List[Dict]:
    """
    Run OLS regression for each year pair.
    
    Args:
        db: Database connection
        
    Returns:
        List of regression results by year
    """
    results = []
    
    # Get unique year pairs from database
    year_pairs = db.query("""
        SELECT DISTINCT year_from, year_to 
        FROM player_deltas 
        ORDER BY year_from
    """)
    
    for year_from, year_to in year_pairs:
        # Skip if we don't have next year data (can't validate)
        next_year = year_to + 1
        
        # Get data for this year pair
        data = get_regression_data(db, year_from=year_from, year_to=year_to)
        
        if len(data) < 10:
            results.append({
                'year_pair': f"{year_from}-{year_to}",
                'n': len(data),
                'error': 'Insufficient data'
            })
            continue
        
        # Run regression
        try:
            reg_result = run_ols_regression(data)
            reg_result['year_pair'] = f"{year_from}-{year_to}"
            results.append(reg_result)
        except Exception as e:
            results.append({
                'year_pair': f"{year_from}-{year_to}",
                'error': str(e)
            })
    
    return results


def run_pooled_regression(db: MadduxDatabase) -> Dict:
    """
    Run pooled OLS regression across all years.
    
    Args:
        db: Database connection
        
    Returns:
        Regression results dictionary
    """
    # Get all data
    data = get_regression_data(db)
    
    if len(data) < 30:
        return {'error': 'Insufficient data for pooled regression', 'n': len(data)}
    
    # Run regression
    result = run_ols_regression(data)
    result['type'] = 'pooled'
    result['years'] = f"{data['year_from'].min()}-{data['year_to'].max()}"
    
    return result


def calculate_maddux_correlation(db: MadduxDatabase, target: str = 'ops_change') -> Dict:
    """
    Calculate correlation between MADDUX score and specified target.
    
    This is the key validation metric.
    
    Args:
        db: Database connection
        target: Target variable ('ops_change', 'wrc_plus_change')
        
    Returns:
        Dict with correlation statistics
    """
    data = get_regression_data(db, target=target)
    
    if len(data) < 10:
        return {'error': 'Insufficient data', 'n': len(data)}
    
    # Calculate correlation
    maddux_scores = data['maddux_score'].values
    target_values = data[target].values
    
    # Remove NaN
    mask = ~(np.isnan(maddux_scores) | np.isnan(target_values))
    maddux_scores = maddux_scores[mask]
    target_values = target_values[mask]
    
    if len(maddux_scores) < 10:
        return {'error': 'Insufficient valid data', 'n': len(maddux_scores)}
    
    correlation = np.corrcoef(maddux_scores, target_values)[0, 1]
    
    # Statistical significance
    _, p_value = stats.pearsonr(maddux_scores, target_values)
    
    return {
        'n': len(maddux_scores),
        'target': target,
        'correlation': correlation,
        'r_squared': correlation ** 2,
        'p_value': p_value,
        'target_correlation': 0.65 if target == 'ops_change' else 0.60,
        'meets_target': correlation > 0.65 if target == 'ops_change' else correlation > 0.60
    }


def run_multi_target_regression(db: MadduxDatabase) -> Dict[str, Dict]:
    """
    Run regression for all target variables.
    
    Args:
        db: Database connection
        
    Returns:
        Dict with results for each target
    """
    results = {}
    
    for target_key, target_info in TARGET_VARIABLES.items():
        if target_info['type'] == 'regression':
            # Run OLS regression
            data = get_regression_data(db, target=target_key)
            if len(data) >= 30:
                result = run_ols_regression_target(data, target_key)
                results[target_key] = result
        elif target_info['type'] == 'classification' and SKLEARN_AVAILABLE:
            # Run logistic regression
            data = get_regression_data(db)
            if len(data) >= 30:
                result = run_classification(data)
                results[target_key] = result
    
    return results


def run_ols_regression_target(data: pd.DataFrame, target: str = 'ops_change') -> Dict:
    """
    Run OLS regression for specified target variable.
    
    Args:
        data: DataFrame with regression variables
        target: Target column name
        
    Returns:
        Dict with regression results
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for OLS regression")
    
    if len(data) < 10 or target not in data.columns:
        return {'error': f'Insufficient data or invalid target: {target}', 'n': len(data)}
    
    # Prepare data
    X = data[['delta_max_ev', 'delta_hard_hit_pct']].values
    y = data[target].values
    
    # Remove any NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    if len(X) < 10:
        return {'error': 'Insufficient valid data', 'n': len(X)}
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    # Calculate additional statistics
    y_pred = results.predict(X_with_const)
    correlation = np.corrcoef(y, y_pred)[0, 1]
    mae = np.mean(np.abs(y - y_pred))
    
    return {
        'target': target,
        'n': len(X),
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'correlation': correlation,
        'mae': mae,
        'intercept': results.params[0],
        'beta_max_ev': results.params[1],
        'beta_hard_hit': results.params[2],
        'p_max_ev': results.pvalues[1],
        'p_hard_hit': results.pvalues[2],
        'f_statistic': results.fvalue,
        'f_pvalue': results.f_pvalue
    }


def run_classification(data: pd.DataFrame) -> Dict:
    """
    Run logistic regression for binary classification target.
    
    Args:
        data: DataFrame with features and improved_binary column
        
    Returns:
        Dict with classification metrics
    """
    if not SKLEARN_AVAILABLE:
        return {'error': 'sklearn required for classification'}
    
    if 'improved_binary' not in data.columns:
        return {'error': 'improved_binary column not found'}
    
    # Prepare data
    X = data[['delta_max_ev', 'delta_hard_hit_pct', 'maddux_score']].values
    y = data['improved_binary'].values
    
    # Remove NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    if len(X) < 30:
        return {'error': 'Insufficient data', 'n': len(X)}
    
    # Fit logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    return {
        'target': 'improved_binary',
        'n': len(X),
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y, y_prob),
        'baseline_rate': y.mean(),  # Percentage who improved
        'coefficients': {
            'delta_max_ev': model.coef_[0][0],
            'delta_hard_hit_pct': model.coef_[0][1],
            'maddux_score': model.coef_[0][2]
        }
    }


def run_feature_regression(db: MadduxDatabase, target: str = 'ops_change') -> Dict:
    """
    Run regression using engineered features instead of raw deltas.
    
    Args:
        db: Database connection
        target: Target variable
        
    Returns:
        Dict with regression results
    """
    if not STATSMODELS_AVAILABLE:
        return {'error': 'statsmodels required'}
    
    data = get_feature_regression_data(db)
    
    if len(data) < 30:
        return {'error': 'Insufficient data', 'n': len(data)}
    
    # Feature columns
    feature_cols = [
        'underperformance_gap', 'deviation_from_baseline', 'career_peak_deviation',
        'improvement_momentum', 'combined_luck_index', 'quality_zscore',
        'hard_hit_zscore', 'age_factor'
    ]
    
    available_cols = [c for c in feature_cols if c in data.columns]
    
    if len(available_cols) < 3:
        return {'error': 'Insufficient feature columns', 'available': available_cols}
    
    # Prepare data
    X = data[available_cols].fillna(0).values
    y = data[target].values
    
    # Remove NaN in target
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    if len(X) < 30:
        return {'error': 'Insufficient valid data', 'n': len(X)}
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    # Calculate metrics
    y_pred = results.predict(X_with_const)
    correlation = np.corrcoef(y, y_pred)[0, 1]
    mae = np.mean(np.abs(y - y_pred))
    
    # Feature coefficients
    coefficients = dict(zip(available_cols, results.params[1:]))
    p_values = dict(zip(available_cols, results.pvalues[1:]))
    
    return {
        'target': target,
        'n': len(X),
        'n_features': len(available_cols),
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'correlation': correlation,
        'mae': mae,
        'coefficients': coefficients,
        'p_values': p_values,
        'f_statistic': results.fvalue,
        'f_pvalue': results.f_pvalue,
        'features_used': available_cols
    }


def format_regression_results(results: Dict) -> str:
    """Format regression results for display."""
    if 'error' in results:
        return f"Error: {results['error']}"
    
    lines = [
        f"Sample size (n): {results['n']}",
        f"R-squared: {results['r_squared']:.4f}",
        f"Adjusted R-squared: {results['adj_r_squared']:.4f}",
        f"Correlation (r): {results['correlation']:.4f}",
        f"Mean Absolute Error: {results['mae']:.4f}",
        "",
        "Coefficients:",
        f"  Intercept: {results['intercept']:.6f}",
        f"  β Max EV: {results['beta_max_ev']:.6f} (p={results['p_max_ev']:.4f})",
        f"  β Hard Hit%: {results['beta_hard_hit']:.6f} (p={results['p_hard_hit']:.4f})",
        "",
        f"Derived Ratio (β_HH / β_EV): {results['derived_ratio']:.2f}" if results['derived_ratio'] else "Derived Ratio: N/A",
        f"Proposed Ratio: {results['proposed_ratio']:.1f}",
        "",
        f"F-statistic: {results['f_statistic']:.2f} (p={results['f_pvalue']:.6f})"
    ]
    
    return "\n".join(lines)


def main():
    """Run all regression analyses with multi-target support."""
    print("=" * 70)
    print("MADDUX™ Algorithm - OLS Regression Validation (Enhanced v2)")
    print("=" * 70)
    
    if not STATSMODELS_AVAILABLE:
        print("Error: statsmodels not installed")
        return
    
    db = MadduxDatabase()
    
    # 1. Original MADDUX formula validation
    print("\n1. ORIGINAL MADDUX FORMULA (OPS Target)")
    print("-" * 70)
    pooled = run_pooled_regression(db)
    print(format_regression_results(pooled))
    
    # 2. Multi-target analysis
    print("\n2. MULTI-TARGET REGRESSION ANALYSIS")
    print("-" * 70)
    
    for target_key, target_info in TARGET_VARIABLES.items():
        print(f"\n  {target_info['name']} ({target_key})")
        print(f"  {target_info['description']}")
        print("  " + "-" * 50)
        
        if target_info['type'] == 'regression':
            result = run_ols_regression_target(get_regression_data(db), target_key)
            if 'error' not in result:
                print(f"  n={result['n']}, R²={result['r_squared']:.4f}, "
                      f"r={result['correlation']:.4f}, MAE={result['mae']:.4f}")
            else:
                print(f"  Error: {result['error']}")
        
        elif target_info['type'] == 'classification' and SKLEARN_AVAILABLE:
            result = run_classification(get_regression_data(db))
            if 'error' not in result:
                print(f"  n={result['n']}, Accuracy={result['accuracy']:.3f}, "
                      f"Precision={result['precision']:.3f}, AUC={result['auc_roc']:.3f}")
                print(f"  Baseline rate (% improved): {result['baseline_rate']*100:.1f}%")
            else:
                print(f"  Error: {result['error']}")
    
    # 3. Feature-based regression (using engineered features)
    print("\n3. FEATURE-BASED REGRESSION (Engineered Features)")
    print("-" * 70)
    
    for target in ['ops_change', 'wrc_plus_change']:
        result = run_feature_regression(db, target)
        if 'error' not in result:
            print(f"\n  Target: {target}")
            print(f"  n={result['n']}, R²={result['r_squared']:.4f}, "
                  f"r={result['correlation']:.4f}")
            print(f"  Features used: {', '.join(result['features_used'][:5])}...")
            
            # Top coefficients
            if 'coefficients' in result:
                sorted_coef = sorted(result['coefficients'].items(), 
                                    key=lambda x: abs(x[1]), reverse=True)[:3]
                print(f"  Top predictors:")
                for feat, coef in sorted_coef:
                    p_val = result['p_values'].get(feat, 1.0)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"    {feat}: {coef:+.4f} {sig}")
        else:
            print(f"\n  Target: {target} - Error: {result['error']}")
    
    # 4. Correlation comparison
    print("\n4. MADDUX SCORE CORRELATION BY TARGET")
    print("-" * 70)
    
    print(f"{'Target':<20} {'n':>8} {'r':>10} {'p-value':>12} {'Meets':>8}")
    print("-" * 70)
    
    for target in ['ops_change', 'wrc_plus_change']:
        corr = calculate_maddux_correlation(db, target=target)
        if 'error' not in corr:
            meets = "YES" if corr['meets_target'] else "NO"
            print(f"{target:<20} {corr['n']:>8} {corr['correlation']:>+10.4f} "
                  f"{corr['p_value']:>12.6f} {meets:>8}")
        else:
            print(f"{target:<20} Error: {corr['error']}")
    
    # 5. Yearly breakdown
    print("\n5. YEARLY REGRESSIONS (OPS)")
    print("-" * 70)
    yearly = run_yearly_regressions(db)
    
    print(f"{'Year Pair':<12} {'N':>6} {'R²':>8} {'r':>8} {'Derived':>10}")
    print("-" * 50)
    
    for result in yearly:
        if 'error' in result:
            print(f"{result['year_pair']:<12} {result.get('n', 'N/A'):>6} {'Error':>8}")
        else:
            derived = f"{result['derived_ratio']:.2f}" if result.get('derived_ratio') else "N/A"
            print(f"{result['year_pair']:<12} {result['n']:>6} {result['r_squared']:>8.4f} "
                  f"{result['correlation']:>8.4f} {derived:>10}")
    
    # 6. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if 'error' not in pooled:
        print(f"\nOriginal MADDUX Formula Performance:")
        print(f"  R-squared: {pooled['r_squared']:.4f} ({pooled['r_squared']*100:.1f}% variance explained)")
        print(f"  Correlation: {pooled['correlation']:.4f}")
        
        target_met = pooled['correlation'] >= 0.65
        print(f"  Target (r > 0.65): {'MET ✓' if target_met else 'NOT MET ✗'}")
        
        if not target_met:
            print("\n  ⚠️  Original formula shows weak correlation!")
            print("  Consider using engineered features and stacking model.")
    
    print("\nRecommendation:")
    feature_result = run_feature_regression(db, 'ops_change')
    if 'error' not in feature_result and feature_result['correlation'] > pooled.get('correlation', 0):
        print(f"  Feature-based model shows improved r={feature_result['correlation']:.4f}")
        print("  Use stacking_model.py for best predictions.")
    else:
        print("  Run feature_engineering.py to calculate features, then re-run.")
    
    db.close()


if __name__ == "__main__":
    main()

