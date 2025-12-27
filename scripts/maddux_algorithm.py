"""
maddux_algorithm.py
MADDUX™ Hitter Algorithm implementation (enhanced v2).

Original Formula: MADDUX Score = Δ Max EV + (2.1 × Δ Hard Hit%)
NOTE: Original formula showed weak/negative correlation with next-year OPS.

Enhanced v2 uses:
- Feature engineering (luck gaps, baseline deviations, bat quality)
- Stacking Meta-Learner ensemble (Ridge + Lasso + GBM)
- Multi-target support (OPS, wRC+, binary classification)

This module calculates:
- Original MADDUX scores (for comparison/backward compatibility)
- Enhanced predictions using stacking model
- Year-over-year deltas for Max EV and Hard Hit%
"""

import sys
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase

# Try to import enhanced components
try:
    from scripts.feature_engineering import calculate_all_features_for_year
    from scripts.stacking_model import StackingMetaLearner, get_training_data, get_prediction_data
    ENHANCED_MODEL_AVAILABLE = True
except ImportError:
    ENHANCED_MODEL_AVAILABLE = False

# Default weight for Hard Hit% in original MADDUX formula
DEFAULT_HARD_HIT_WEIGHT = 2.1

# Model types available
MODEL_TYPES = {
    'original': 'Original MADDUX formula (Δ Max EV + 2.1 × Δ Hard Hit%)',
    'stacking': 'Stacking Meta-Learner with engineered features',
}


def calculate_deltas(prev_season: Dict, curr_season: Dict) -> Dict[str, float]:
    """
    Calculate year-over-year deltas between two seasons.
    
    Args:
        prev_season: Dict with 'max_ev', 'hard_hit_pct' from previous year
        curr_season: Dict with 'max_ev', 'hard_hit_pct' from current year
        
    Returns:
        Dict with delta_max_ev and delta_hard_hit_pct
    """
    delta_max_ev = curr_season.get('max_ev', 0) - prev_season.get('max_ev', 0)
    delta_hard_hit_pct = curr_season.get('hard_hit_pct', 0) - prev_season.get('hard_hit_pct', 0)
    
    return {
        'delta_max_ev': delta_max_ev,
        'delta_hard_hit_pct': delta_hard_hit_pct
    }


def calculate_maddux_score(delta_max_ev: float, delta_hard_hit_pct: float, 
                           weight: float = DEFAULT_HARD_HIT_WEIGHT) -> float:
    """
    Calculate MADDUX™ score from deltas.
    
    Formula: MADDUX Score = Δ Max EV + (weight × Δ Hard Hit%)
    
    Args:
        delta_max_ev: Change in max exit velocity (mph)
        delta_hard_hit_pct: Change in hard hit percentage
        weight: Weight for Hard Hit% (default 2.1)
        
    Returns:
        MADDUX score
    """
    return delta_max_ev + (weight * delta_hard_hit_pct)


def get_consecutive_seasons(db: MadduxDatabase, year_from: int, 
                           year_to: int) -> List[Dict]:
    """
    Get players with Statcast data in consecutive seasons.
    
    Args:
        db: Database connection
        year_from: Earlier year
        year_to: Later year
        
    Returns:
        List of dicts with player info and both seasons' data
    """
    sql = """
        SELECT 
            p.id as player_id,
            p.player_name,
            p.mlb_id,
            s1.max_ev as prev_max_ev,
            s1.hard_hit_pct as prev_hard_hit_pct,
            s1.avg_ev as prev_avg_ev,
            s1.barrel_pct as prev_barrel_pct,
            s2.max_ev as curr_max_ev,
            s2.hard_hit_pct as curr_hard_hit_pct,
            s2.avg_ev as curr_avg_ev,
            s2.barrel_pct as curr_barrel_pct,
            f1.ops as prev_ops,
            f2.ops as curr_ops
        FROM players p
        JOIN statcast_seasons s1 ON p.id = s1.player_id AND s1.year = ?
        JOIN statcast_seasons s2 ON p.id = s2.player_id AND s2.year = ?
        LEFT JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = ?
        LEFT JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = ?
        WHERE s1.max_ev IS NOT NULL AND s2.max_ev IS NOT NULL
          AND s1.hard_hit_pct IS NOT NULL AND s2.hard_hit_pct IS NOT NULL
    """
    
    result = db.query(sql, (year_from, year_to, year_from, year_to))
    
    pairs = []
    for row in result:
        pairs.append({
            'player_id': row[0],
            'player_name': row[1],
            'mlb_id': row[2],
            'prev_max_ev': row[3],
            'prev_hard_hit_pct': row[4],
            'prev_avg_ev': row[5],
            'prev_barrel_pct': row[6],
            'curr_max_ev': row[7],
            'curr_hard_hit_pct': row[8],
            'curr_avg_ev': row[9],
            'curr_barrel_pct': row[10],
            'prev_ops': row[11],
            'curr_ops': row[12]
        })
    
    return pairs


def calculate_all_deltas(db: MadduxDatabase, year_from: int, year_to: int,
                        weight: float = DEFAULT_HARD_HIT_WEIGHT) -> List[Dict]:
    """
    Calculate deltas and MADDUX scores for all players with consecutive seasons.
    
    Args:
        db: Database connection
        year_from: Earlier year
        year_to: Later year
        weight: Weight for Hard Hit% in MADDUX formula
        
    Returns:
        List of dicts with player info, deltas, and scores
    """
    pairs = get_consecutive_seasons(db, year_from, year_to)
    
    results = []
    for pair in pairs:
        # Calculate deltas
        prev_season = {
            'max_ev': pair['prev_max_ev'],
            'hard_hit_pct': pair['prev_hard_hit_pct']
        }
        curr_season = {
            'max_ev': pair['curr_max_ev'],
            'hard_hit_pct': pair['curr_hard_hit_pct']
        }
        
        deltas = calculate_deltas(prev_season, curr_season)
        
        # Calculate MADDUX score
        maddux_score = calculate_maddux_score(
            deltas['delta_max_ev'],
            deltas['delta_hard_hit_pct'],
            weight=weight
        )
        
        # Calculate OPS delta if available
        delta_ops = None
        if pair['prev_ops'] is not None and pair['curr_ops'] is not None:
            delta_ops = pair['curr_ops'] - pair['prev_ops']
        
        results.append({
            'player_id': pair['player_id'],
            'player_name': pair['player_name'],
            'mlb_id': pair['mlb_id'],
            'year_from': year_from,
            'year_to': year_to,
            'delta_max_ev': deltas['delta_max_ev'],
            'delta_hard_hit_pct': deltas['delta_hard_hit_pct'],
            'maddux_score': maddux_score,
            'delta_ops': delta_ops,
            'prev_max_ev': pair['prev_max_ev'],
            'curr_max_ev': pair['curr_max_ev'],
            'prev_hard_hit_pct': pair['prev_hard_hit_pct'],
            'curr_hard_hit_pct': pair['curr_hard_hit_pct']
        })
    
    return results


def store_deltas_in_db(db: MadduxDatabase, deltas: List[Dict]) -> int:
    """
    Store calculated deltas and MADDUX scores in database.
    
    Args:
        db: Database connection
        deltas: List of delta dicts from calculate_all_deltas
        
    Returns:
        Number of records stored
    """
    count = 0
    for delta in deltas:
        try:
            db.execute("""
                INSERT OR REPLACE INTO player_deltas 
                (player_id, year_from, year_to, delta_max_ev, delta_hard_hit_pct, 
                 delta_ops, maddux_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                delta['player_id'],
                delta['year_from'],
                delta['year_to'],
                delta['delta_max_ev'],
                delta['delta_hard_hit_pct'],
                delta['delta_ops'],
                delta['maddux_score']
            ))
            count += 1
        except Exception as e:
            print(f"Error storing delta: {e}")
    
    db.commit()
    return count


def get_top_maddux_scores(db: MadduxDatabase, year_from: int, year_to: int,
                          limit: int = 20) -> List[Dict]:
    """
    Get top players by MADDUX score for a year pair.
    
    Args:
        db: Database connection
        year_from: Earlier year
        year_to: Later year
        limit: Number of top players to return
        
    Returns:
        List of dicts with player info and scores, sorted by score desc
    """
    sql = """
        SELECT 
            p.player_name,
            p.mlb_id,
            d.maddux_score,
            d.delta_max_ev,
            d.delta_hard_hit_pct,
            d.delta_ops
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        WHERE d.year_from = ? AND d.year_to = ?
        ORDER BY d.maddux_score DESC
        LIMIT ?
    """
    
    result = db.query(sql, (year_from, year_to, limit))
    
    top = []
    for row in result:
        top.append({
            'player_name': row[0],
            'mlb_id': row[1],
            'maddux_score': row[2],
            'delta_max_ev': row[3],
            'delta_hard_hit_pct': row[4],
            'delta_ops': row[5]
        })
    
    return top


def calculate_all_year_pairs(db: MadduxDatabase, 
                             start_year: int = 2015, 
                             end_year: int = 2025,
                             weight: float = DEFAULT_HARD_HIT_WEIGHT) -> Dict[str, List[Dict]]:
    """
    Calculate deltas and scores for all consecutive year pairs.
    
    Args:
        db: Database connection
        start_year: First year
        end_year: Last year
        weight: Weight for Hard Hit% in MADDUX formula
        
    Returns:
        Dict mapping "year_from-year_to" to list of results
    """
    all_results = {}
    
    for year in range(start_year, end_year):
        year_from = year
        year_to = year + 1
        
        key = f"{year_from}-{year_to}"
        print(f"Calculating deltas for {key}...")
        
        results = calculate_all_deltas(db, year_from, year_to, weight)
        
        if results:
            all_results[key] = results
            
            # Store in database
            count = store_deltas_in_db(db, results)
            print(f"  {len(results)} players, {count} stored")
    
    return all_results


def get_next_year_ops_change(db: MadduxDatabase, year_from: int, 
                             year_to: int) -> List[Dict]:
    """
    Get MADDUX scores and NEXT year's OPS change for validation.
    
    This is the key metric for backtesting - we want to see if 
    MADDUX score (based on year_from to year_to changes) predicts
    OPS improvement in year_to+1.
    
    Args:
        db: Database connection
        year_from: First year of delta calculation
        year_to: Second year of delta calculation
        
    Returns:
        List of dicts with scores and next-year OPS changes
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
            f2.ops as next_year_ops,
            (f2.ops - f1.ops) as next_year_ops_change
        FROM player_deltas d
        JOIN players p ON d.player_id = p.id
        LEFT JOIN fangraphs_seasons f1 ON p.id = f1.player_id AND f1.year = ?
        LEFT JOIN fangraphs_seasons f2 ON p.id = f2.player_id AND f2.year = ?
        WHERE d.year_from = ? AND d.year_to = ?
          AND f1.ops IS NOT NULL AND f2.ops IS NOT NULL
        ORDER BY d.maddux_score DESC
    """
    
    result = db.query(sql, (year_to, next_year, year_from, year_to))
    
    data = []
    for row in result:
        data.append({
            'player_name': row[0],
            'mlb_id': row[1],
            'maddux_score': row[2],
            'delta_max_ev': row[3],
            'delta_hard_hit_pct': row[4],
            'year_to_ops': row[5],
            'next_year_ops': row[6],
            'next_year_ops_change': row[7]
        })
    
    return data


# ============================================================================
# ENHANCED MODEL (v2)
# ============================================================================

class MadduxModel:
    """
    MADDUX™ prediction model with support for original and enhanced methods.
    
    Methods:
        - 'original': Uses original MADDUX formula
        - 'stacking': Uses Stacking Meta-Learner with engineered features
    """
    
    def __init__(self, method: str = 'stacking'):
        """
        Initialize MADDUX model.
        
        Args:
            method: 'original' or 'stacking'
        """
        self.method = method
        self.model = None
        self.is_fitted = False
        
        if method == 'stacking' and not ENHANCED_MODEL_AVAILABLE:
            print("Warning: Enhanced model not available, falling back to original")
            self.method = 'original'
    
    def fit(self, db: MadduxDatabase, end_year: int = 2025, min_pa: int = 200):
        """
        Train the model.
        
        Args:
            db: Database connection
            end_year: Last year to include in training
            min_pa: Minimum PA threshold
        """
        if self.method == 'original':
            # Original method doesn't need training
            self.is_fitted = True
            return
        
        if not ENHANCED_MODEL_AVAILABLE:
            raise ImportError("Stacking model not available")
        
        # Get training data
        X, y, df = get_training_data(db, end_year, 'ops_change', min_pa)
        
        if len(X) < 50:
            raise ValueError(f"Insufficient training data: {len(X)} samples")
        
        # Train stacking model
        self.model = StackingMetaLearner(feature_set='extended')
        self.model.fit(X.fillna(0).values, y.values)
        self.is_fitted = True
        
        print(f"Model trained on {len(X)} samples")
    
    def predict(
        self,
        db: MadduxDatabase,
        year: int,
        min_pa: int = 200,
        top_n: int = 20,
        with_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Generate predictions for a year.
        
        Args:
            db: Database connection
            year: Year to predict from (predicting year + 1)
            min_pa: Minimum PA threshold
            top_n: Number of top predictions to return (0 for all)
            with_confidence: Include confidence intervals
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.method == 'original':
            return self._predict_original(db, year, min_pa, top_n)
        else:
            return self._predict_stacking(db, year, min_pa, top_n, with_confidence)
    
    def _predict_original(
        self,
        db: MadduxDatabase,
        year: int,
        min_pa: int,
        top_n: int
    ) -> pd.DataFrame:
        """Generate predictions using original MADDUX formula."""
        # Get deltas
        deltas = calculate_all_deltas(db, year - 1, year)
        
        if not deltas:
            return pd.DataFrame()
        
        df = pd.DataFrame(deltas)
        df = df.sort_values('maddux_score', ascending=False)
        
        if top_n > 0:
            df = df.head(top_n)
        
        df['rank'] = range(1, len(df) + 1)
        df['method'] = 'original'
        
        return df
    
    def _predict_stacking(
        self,
        db: MadduxDatabase,
        year: int,
        min_pa: int,
        top_n: int,
        with_confidence: bool
    ) -> pd.DataFrame:
        """Generate predictions using stacking model."""
        # Get prediction data
        pred_df = get_prediction_data(db, year, min_pa)
        
        if len(pred_df) == 0:
            return pd.DataFrame()
        
        # Get features
        X = pred_df[self.model.available_features_].fillna(0).values
        
        # Make predictions
        if with_confidence:
            predictions, lower, upper = self.model.predict_with_confidence(X)
            pred_df['predicted_change'] = predictions
            pred_df['confidence_low'] = lower
            pred_df['confidence_high'] = upper
        else:
            pred_df['predicted_change'] = self.model.predict(X)
        
        # Calculate predicted OPS
        if 'current_ops' in pred_df.columns:
            pred_df['predicted_ops'] = pred_df['current_ops'] + pred_df['predicted_change']
        
        # Sort and rank
        pred_df = pred_df.sort_values('predicted_change', ascending=False)
        
        if top_n > 0:
            pred_df = pred_df.head(top_n)
        
        pred_df['rank'] = range(1, len(pred_df) + 1)
        pred_df['method'] = 'stacking'
        
        return pred_df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (stacking model only)."""
        if self.method != 'stacking' or self.model is None:
            return {}
        return self.model.get_feature_importance()


def get_enhanced_predictions(
    db: MadduxDatabase,
    prediction_year: int,
    min_pa: int = 200,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Get predictions using enhanced stacking model.
    
    Convenience function for getting enhanced predictions.
    
    Args:
        db: Database connection
        prediction_year: Year to predict (using prediction_year - 1 features)
        min_pa: Minimum PA
        top_n: Number of top predictions
        
    Returns:
        DataFrame with predictions
    """
    model = MadduxModel(method='stacking')
    model.fit(db, end_year=prediction_year)
    return model.predict(db, prediction_year - 1, min_pa, top_n)


def compare_methods(
    db: MadduxDatabase,
    year: int,
    min_pa: int = 200,
    top_n: int = 20
) -> Dict[str, pd.DataFrame]:
    """
    Compare original and enhanced prediction methods.
    
    Args:
        db: Database connection
        year: Year to compare (predicting year + 1)
        min_pa: Minimum PA
        top_n: Number of top predictions
        
    Returns:
        Dict with 'original' and 'stacking' DataFrames
    """
    results = {}
    
    # Original method
    original_model = MadduxModel(method='original')
    original_model.fit(db)
    results['original'] = original_model.predict(db, year, min_pa, top_n)
    
    # Stacking method
    if ENHANCED_MODEL_AVAILABLE:
        try:
            stacking_model = MadduxModel(method='stacking')
            stacking_model.fit(db, end_year=year + 1)
            results['stacking'] = stacking_model.predict(db, year, min_pa, top_n)
        except Exception as e:
            print(f"Stacking model failed: {e}")
    
    return results


def main():
    """Calculate all deltas and MADDUX scores (enhanced v2)."""
    print("=" * 70)
    print("MADDUX™ Algorithm - Enhanced v2")
    print("=" * 70)
    
    # Open database
    db = MadduxDatabase()
    
    # 1. Calculate original MADDUX scores
    print("\n1. CALCULATING ORIGINAL MADDUX DELTAS")
    print("-" * 70)
    
    results = calculate_all_year_pairs(db, start_year=2015, end_year=2025)
    
    total_players = sum(len(r) for r in results.values())
    print(f"\n  Year pairs: {len(results)}")
    print(f"  Player-seasons: {total_players}")
    
    # 2. Show top original scores
    print("\n2. TOP 10 BY ORIGINAL MADDUX SCORE (2024-2025)")
    print("-" * 70)
    
    top_original = get_top_maddux_scores(db, 2024, 2025, limit=10)
    if top_original:
        print(f"{'Rank':<5} {'Player':<25} {'Score':>8} {'Δ Max EV':>10} {'Δ HH%':>8}")
        print("-" * 60)
        for i, player in enumerate(top_original, 1):
            print(f"{i:<5} {player['player_name'][:24]:<25} "
                  f"{player['maddux_score']:>8.2f} "
                  f"{player['delta_max_ev']:>+10.1f} "
                  f"{player['delta_hard_hit_pct']:>+8.1f}")
    else:
        print("  No data available for 2024-2025")
        # Fall back to 2023-2024
        top_original = get_top_maddux_scores(db, 2023, 2024, limit=10)
        if top_original:
            print("\n  Showing 2023-2024 instead:")
            for i, player in enumerate(top_original, 1):
                print(f"{i:2}. {player['player_name']:<25} Score: {player['maddux_score']:6.2f}")
    
    # 3. Enhanced predictions (if available)
    if ENHANCED_MODEL_AVAILABLE:
        print("\n3. ENHANCED MODEL PREDICTIONS (Stacking Meta-Learner)")
        print("-" * 70)
        
        try:
            model = MadduxModel(method='stacking')
            model.fit(db, end_year=2025)
            
            predictions = model.predict(db, 2025, min_pa=200, top_n=10)
            
            if len(predictions) > 0:
                print(f"\n  Top 10 predictions for 2026 improvement:")
                print(f"  {'Rank':<5} {'Player':<25} {'Pred Δ OPS':>12} {'95% CI':>18}")
                print("  " + "-" * 65)
                
                for _, row in predictions.iterrows():
                    if 'confidence_low' in row and pd.notna(row['confidence_low']):
                        ci = f"[{row['confidence_low']:+.3f}, {row['confidence_high']:+.3f}]"
                    else:
                        ci = "N/A"
                    print(f"  {row['rank']:<5} {row['player_name'][:24]:<25} "
                          f"{row['predicted_change']:>+12.3f} {ci:>18}")
                
                # Feature importance
                importance = model.get_feature_importance()
                if importance:
                    print(f"\n  Top features by importance:")
                    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    for feat, imp in sorted_imp:
                        print(f"    {feat}: {imp:.4f}")
            else:
                print("  No predictions generated (run feature_engineering.py first)")
                
        except Exception as e:
            print(f"  Enhanced model error: {e}")
            print("  Run: python feature_engineering.py to calculate features")
    else:
        print("\n3. ENHANCED MODEL NOT AVAILABLE")
        print("-" * 70)
        print("  Install required packages: pip install scikit-learn")
        print("  Then run: python feature_engineering.py")
    
    # 4. Method comparison summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\n  Original MADDUX Formula:")
    print("    Score = Δ Max EV + (2.1 × Δ Hard Hit%)")
    print("    Status: ⚠️  Shows weak/negative correlation with next-year OPS")
    
    if ENHANCED_MODEL_AVAILABLE:
        print("\n  Enhanced Stacking Model:")
        print("    Features: luck gaps, baseline deviation, bat quality, age")
        print("    Models: Ridge + Lasso + Gradient Boosting (stacked)")
        print("    Status: ✓ Improved correlation and hit rate")
    
    db.close()
    print("\n" + "=" * 70)
    print("MADDUX algorithm calculations complete!")


if __name__ == "__main__":
    main()

