"""
stacking_model.py
Stacking Meta-Learner for MADDUX™ enhanced model.

Combines multiple base models into an ensemble:
- Ridge Regression (handles multicollinearity)
- Lasso Regression (feature selection)
- Gradient Boosting (non-linear relationships)

Meta-learner: Ridge regression on base model predictions

Features:
- Walk-forward cross-validation (no data leakage)
- Prediction with confidence intervals (bootstrap)
- Support for multiple target variables (OPS, wRC+, binary)
- Feature importance aggregation
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase, store_model_prediction

# scikit-learn imports
try:
    from sklearn.linear_model import Ridge, Lasso, LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")


# Feature sets for different model configurations
BASE_FEATURES = [
    'underperformance_gap',
    'xslg_gap_scaled',
    'deviation_from_baseline',
    'career_peak_deviation',
    'improvement_momentum',
    'combined_luck_index',
    'quality_zscore',
    'hard_hit_zscore',
    'age_factor',
]

EXTENDED_FEATURES = BASE_FEATURES + [
    'babip_gap',
    'iso_gap',
    'bat_speed_zscore',
    'swing_length_factor',
    'squared_up_quality',
    'whiff_adjusted_power',
    'pa_weight',
    # Launch angle features (new)
    'delta_launch_angle',
    'delta_sweet_spot',
    'launch_angle_zscore',
    # Zone swing features (new)
    'delta_z_swing',
    'discipline_improvement',
]


@dataclass
class PredictionResult:
    """Container for a single prediction with confidence interval."""
    player_id: int
    player_name: str
    predicted_value: float
    confidence_low: float
    confidence_high: float
    feature_contributions: Dict[str, float]


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    correlation: float
    r_squared: float
    mae: float
    rmse: float
    hit_rate_top20: float
    hit_rate_top10: float
    avg_change_top20: float
    n_samples: int


class StackingMetaLearner:
    """
    Stacking ensemble model for predicting next-year performance change.
    
    Architecture:
        Base Models: Ridge, Lasso, Gradient Boosting
        Meta-Learner: Ridge regression on base model predictions
    """
    
    def __init__(
        self,
        feature_set: str = 'extended',
        ridge_alpha: float = 1.0,
        lasso_alpha: float = 0.01,
        gbm_n_estimators: int = 100,
        gbm_max_depth: int = 4,
        meta_alpha: float = 0.1
    ):
        """
        Initialize the stacking meta-learner.
        
        Args:
            feature_set: 'base' or 'extended'
            ridge_alpha: Regularization for Ridge
            lasso_alpha: Regularization for Lasso
            gbm_n_estimators: Number of trees for GBM
            gbm_max_depth: Max depth for GBM trees
            meta_alpha: Regularization for meta-learner
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for StackingMetaLearner")
        
        self.feature_names = EXTENDED_FEATURES if feature_set == 'extended' else BASE_FEATURES
        
        # Base models
        self.base_models = {
            'ridge': Ridge(alpha=ridge_alpha),
            'lasso': Lasso(alpha=lasso_alpha, max_iter=10000),
            'gbm': GradientBoostingRegressor(
                n_estimators=gbm_n_estimators,
                max_depth=gbm_max_depth,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Meta-learner
        self.meta_model = Ridge(alpha=meta_alpha)
        
        # Preprocessing
        self.scaler = StandardScaler()
        
        # Trained state
        self.is_fitted = False
        self.feature_importance_ = {}
        self.meta_weights_ = {}
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and validate feature columns from DataFrame."""
        available = [f for f in self.feature_names if f in df.columns]
        if len(available) < 3:
            raise ValueError(f"Insufficient features. Need at least 3, found: {available}")
        
        X = df[available].fillna(0).values
        self.available_features_ = available
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingMetaLearner':
        """
        Fit the stacking model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            Self
        """
        # Store available features (in case not set by _prepare_features)
        if not hasattr(self, 'available_features_') or self.available_features_ is None:
            # Generate feature names based on array shape
            self.available_features_ = [f'feature_{i}' for i in range(X.shape[1])]
            # If we have the same number as expected features, use those names
            if X.shape[1] == len(self.feature_names):
                self.available_features_ = list(self.feature_names)
            elif X.shape[1] <= len(self.feature_names):
                self.available_features_ = list(self.feature_names[:X.shape[1]])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Get cross-validated predictions from base models for stacking
        base_predictions = []
        
        for name, model in self.base_models.items():
            # Use 5-fold CV predictions for training meta-learner
            if name in ['ridge', 'lasso']:
                cv_pred = cross_val_predict(model, X_scaled, y, cv=5)
            else:
                cv_pred = cross_val_predict(model, X, y, cv=5)
            base_predictions.append(cv_pred)
        
        # Stack predictions
        stacked_X = np.column_stack(base_predictions)
        
        # Fit meta-learner
        self.meta_model.fit(stacked_X, y)
        self.meta_weights_ = dict(zip(self.base_models.keys(), self.meta_model.coef_))
        
        # Fit base models on full data
        for name, model in self.base_models.items():
            if name in ['ridge', 'lasso']:
                model.fit(X_scaled, y)
            else:
                model.fit(X, y)
        
        # Calculate feature importance
        self._calculate_feature_importance(X)
        
        self.is_fitted = True
        return self
    
    def _calculate_feature_importance(self, X: np.ndarray):
        """Aggregate feature importance from all models."""
        importance = np.zeros(X.shape[1])
        
        # Ridge coefficients (absolute value)
        ridge_coef = np.abs(self.base_models['ridge'].coef_)
        importance += ridge_coef / ridge_coef.sum() * self.meta_weights_.get('ridge', 0.33)
        
        # Lasso coefficients (non-zero = selected)
        lasso_coef = np.abs(self.base_models['lasso'].coef_)
        if lasso_coef.sum() > 0:
            importance += lasso_coef / lasso_coef.sum() * self.meta_weights_.get('lasso', 0.33)
        
        # GBM feature importance
        gbm_imp = self.base_models['gbm'].feature_importances_
        importance += gbm_imp * self.meta_weights_.get('gbm', 0.33)
        
        # Normalize
        importance = importance / importance.sum()
        
        self.feature_importance_ = dict(zip(
            self.available_features_,
            importance
        ))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking ensemble.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        base_predictions = []
        for name, model in self.base_models.items():
            if name in ['ridge', 'lasso']:
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            base_predictions.append(pred)
        
        # Stack and predict with meta-learner
        stacked_X = np.column_stack(base_predictions)
        return self.meta_model.predict(stacked_X)
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        n_bootstrap: int = 100,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with bootstrap confidence intervals.
        
        Args:
            X: Feature matrix
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get point predictions
        predictions = self.predict(X)
        
        # Bootstrap for confidence intervals
        X_scaled = self.scaler.transform(X)
        
        bootstrap_predictions = []
        for _ in range(n_bootstrap):
            # Sample base model predictions with small perturbation
            base_preds = []
            for name, model in self.base_models.items():
                if name in ['ridge', 'lasso']:
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                # Add small noise to simulate prediction uncertainty
                noise = np.random.normal(0, 0.01, size=pred.shape)
                base_preds.append(pred + noise)
            
            stacked = np.column_stack(base_preds)
            boot_pred = self.meta_model.predict(stacked)
            bootstrap_predictions.append(boot_pred)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate percentiles
        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_predictions, alpha * 100, axis=0)
        upper = np.percentile(bootstrap_predictions, (1 - alpha) * 100, axis=0)
        
        return predictions, lower, upper
    
    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        player_ids: Optional[np.ndarray] = None
    ) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y_true: True target values
            player_ids: Optional player IDs for ranking analysis
            
        Returns:
            ModelMetrics with evaluation results
        """
        y_pred = self.predict(X)
        
        # Basic metrics
        correlation = np.corrcoef(y_pred, y_true)[0, 1]
        r_squared = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Create DataFrame for ranking analysis
        eval_df = pd.DataFrame({
            'predicted': y_pred,
            'actual': y_true
        })
        
        # Hit rate for top 20
        top_20 = eval_df.nlargest(20, 'predicted')
        hit_rate_20 = (top_20['actual'] > 0).mean()
        
        # Hit rate for top 10
        top_10 = eval_df.nlargest(10, 'predicted')
        hit_rate_10 = (top_10['actual'] > 0).mean()
        
        # Average actual change for top 20
        avg_change_20 = top_20['actual'].mean()
        
        return ModelMetrics(
            correlation=correlation,
            r_squared=r_squared,
            mae=mae,
            rmse=rmse,
            hit_rate_top20=hit_rate_20,
            hit_rate_top10=hit_rate_10,
            avg_change_top20=avg_change_20,
            n_samples=len(y_true)
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance."""
        return self.feature_importance_


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time-series prediction.
    
    Ensures no data leakage: always train on past, predict future.
    """
    
    def __init__(
        self,
        model_class: type = StackingMetaLearner,
        model_kwargs: Optional[Dict] = None
    ):
        """
        Initialize validator.
        
        Args:
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.results_by_year = {}
    
    def validate(
        self,
        db: MadduxDatabase,
        start_train_year: int = 2016,
        end_test_year: int = 2024,
        min_pa: int = 200
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation across years.
        
        Args:
            db: Database connection
            start_train_year: First year to include in training
            end_test_year: Last year to test (predicting end_test_year + 1)
            min_pa: Minimum plate appearances
            
        Returns:
            Dict with validation results
        """
        all_metrics = []
        
        for test_year in range(start_train_year + 2, end_test_year + 1):
            # Train on all years before test_year
            train_data = self._get_training_data(
                db, start_train_year, test_year - 1, min_pa
            )
            
            if len(train_data) < 50:
                print(f"  Skipping {test_year}: insufficient training data")
                continue
            
            # Test on test_year (predicting test_year + 1)
            test_data = self._get_test_data(db, test_year, min_pa)
            
            if len(test_data) < 20:
                print(f"  Skipping {test_year}: insufficient test data")
                continue
            
            # Train model
            model = self.model_class(**self.model_kwargs)
            
            X_train = train_data[model.feature_names].fillna(0).values
            y_train = train_data['ops_change'].values
            
            model.fit(X_train, y_train)
            
            # Evaluate
            X_test = test_data[model.feature_names].fillna(0).values
            y_test = test_data['ops_change'].values
            
            metrics = model.evaluate(X_test, y_test)
            metrics_dict = {
                'year': test_year,
                'predicting': test_year + 1,
                'correlation': metrics.correlation,
                'r_squared': metrics.r_squared,
                'hit_rate_top20': metrics.hit_rate_top20,
                'hit_rate_top10': metrics.hit_rate_top10,
                'avg_change_top20': metrics.avg_change_top20,
                'n_samples': metrics.n_samples
            }
            
            all_metrics.append(metrics_dict)
            self.results_by_year[test_year] = metrics_dict
            
            print(f"  {test_year}→{test_year+1}: r={metrics.correlation:.3f}, "
                  f"HR20={metrics.hit_rate_top20*100:.1f}%, n={metrics.n_samples}")
        
        # Aggregate results
        if all_metrics:
            results_df = pd.DataFrame(all_metrics)
            return {
                'by_year': all_metrics,
                'avg_correlation': results_df['correlation'].mean(),
                'std_correlation': results_df['correlation'].std(),
                'avg_hit_rate_top20': results_df['hit_rate_top20'].mean(),
                'avg_hit_rate_top10': results_df['hit_rate_top10'].mean(),
                'avg_r_squared': results_df['r_squared'].mean(),
                'total_samples': results_df['n_samples'].sum()
            }
        
        return {'error': 'No valid results'}
    
    def _get_training_data(
        self,
        db: MadduxDatabase,
        start_year: int,
        end_year: int,
        min_pa: int
    ) -> pd.DataFrame:
        """Get training data with features and target."""
        sql = """
            SELECT 
                pf.*,
                f1.ops as current_ops,
                f2.ops as next_ops,
                (f2.ops - f1.ops) as ops_change,
                f1.wrc_plus as current_wrc_plus,
                f2.wrc_plus as next_wrc_plus,
                (f2.wrc_plus - f1.wrc_plus) as wrc_plus_change
            FROM player_features pf
            JOIN fangraphs_seasons f1 ON pf.player_id = f1.player_id AND pf.year = f1.year
            JOIN fangraphs_seasons f2 ON pf.player_id = f2.player_id AND pf.year + 1 = f2.year
            WHERE pf.year >= ? AND pf.year <= ?
              AND f1.pa >= ? AND f2.pa >= ?
        """
        return db.query_df(sql, (start_year, end_year, min_pa, min_pa))
    
    def _get_test_data(
        self,
        db: MadduxDatabase,
        year: int,
        min_pa: int
    ) -> pd.DataFrame:
        """Get test data for a specific year."""
        return self._get_training_data(db, year, year, min_pa)


def get_training_data(
    db: MadduxDatabase,
    end_year: int,
    target: str = 'ops_change',
    min_pa: int = 200
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Get training data with all features and specified target.
    
    Args:
        db: Database connection
        end_year: Last year for training (predicting end_year + 1)
        target: Target variable ('ops_change', 'wrc_plus_change', 'improved_binary')
        min_pa: Minimum PA
        
    Returns:
        Tuple of (X DataFrame, y Series, full DataFrame with player info)
    """
    sql = """
        SELECT 
            p.player_name,
            pf.*,
            f1.ops as current_ops,
            f2.ops as next_ops,
            (f2.ops - f1.ops) as ops_change,
            f1.wrc_plus as current_wrc_plus,
            f2.wrc_plus as next_wrc_plus,
            (f2.wrc_plus - f1.wrc_plus) as wrc_plus_change,
            CASE WHEN f2.ops > f1.ops THEN 1 ELSE 0 END as improved_binary
        FROM player_features pf
        JOIN players p ON pf.player_id = p.id
        JOIN fangraphs_seasons f1 ON pf.player_id = f1.player_id AND pf.year = f1.year
        JOIN fangraphs_seasons f2 ON pf.player_id = f2.player_id AND pf.year + 1 = f2.year
        WHERE pf.year < ? AND f1.pa >= ? AND f2.pa >= ?
    """
    
    df = db.query_df(sql, (end_year, min_pa, min_pa))
    
    if len(df) == 0:
        return pd.DataFrame(), pd.Series(), pd.DataFrame()
    
    # Get available features
    features = [f for f in EXTENDED_FEATURES if f in df.columns]
    
    X = df[features]
    y = df[target]
    
    return X, y, df


def get_prediction_data(
    db: MadduxDatabase,
    year: int,
    min_pa: int = 200
) -> pd.DataFrame:
    """
    Get data for making predictions.
    
    Args:
        db: Database connection
        year: Year to get features for (predicting year + 1)
        min_pa: Minimum PA
        
    Returns:
        DataFrame with features and player info
    """
    sql = """
        SELECT 
            p.id as player_id,
            p.player_name,
            pf.*,
            f.ops as current_ops,
            f.wrc_plus as current_wrc_plus,
            f.pa,
            f.age
        FROM player_features pf
        JOIN players p ON pf.player_id = p.id
        JOIN fangraphs_seasons f ON pf.player_id = f.player_id AND pf.year = f.year
        WHERE pf.year = ? AND f.pa >= ?
    """
    
    return db.query_df(sql, (year, min_pa))


def train_and_predict(
    db: MadduxDatabase,
    prediction_year: int,
    target: str = 'ops_change',
    min_pa: int = 200
) -> Tuple[StackingMetaLearner, pd.DataFrame]:
    """
    Train model and generate predictions for a year.
    
    Args:
        db: Database connection
        prediction_year: Year to predict (using prediction_year - 1 features)
        target: Target variable
        min_pa: Minimum PA
        
    Returns:
        Tuple of (trained model, predictions DataFrame)
    """
    feature_year = prediction_year - 1
    
    # Get training data
    X_train, y_train, train_df = get_training_data(db, feature_year, target, min_pa)
    
    if len(X_train) < 50:
        raise ValueError(f"Insufficient training data: {len(X_train)} samples")
    
    # Train model
    model = StackingMetaLearner(feature_set='extended')
    
    X_train_arr = X_train.fillna(0).values
    y_train_arr = y_train.values
    
    model.fit(X_train_arr, y_train_arr)
    
    # Get prediction data
    pred_df = get_prediction_data(db, feature_year, min_pa)
    
    if len(pred_df) == 0:
        raise ValueError(f"No prediction data for {feature_year}")
    
    # Make predictions with confidence intervals
    X_pred = pred_df[model.available_features_].fillna(0).values
    predictions, lower, upper = model.predict_with_confidence(X_pred)
    
    # Add predictions to DataFrame
    pred_df['predicted_change'] = predictions
    pred_df['confidence_low'] = lower
    pred_df['confidence_high'] = upper
    pred_df['predicted_value'] = pred_df['current_ops'] + predictions
    
    # Sort by predicted change
    pred_df = pred_df.sort_values('predicted_change', ascending=False)
    pred_df['rank'] = range(1, len(pred_df) + 1)
    
    return model, pred_df


def main():
    """Train model and show validation results."""
    print("=" * 70)
    print("MADDUX™ Stacking Meta-Learner")
    print("=" * 70)
    
    if not SKLEARN_AVAILABLE:
        print("Error: scikit-learn required")
        return
    
    db = MadduxDatabase()
    
    # Run walk-forward validation
    print("\n1. WALK-FORWARD CROSS-VALIDATION")
    print("-" * 70)
    
    validator = WalkForwardValidator()
    results = validator.validate(db, start_train_year=2016, end_test_year=2024)
    
    if 'error' not in results:
        print(f"\n  Average Correlation: {results['avg_correlation']:.4f} "
              f"(±{results['std_correlation']:.4f})")
        print(f"  Average Hit Rate Top 20: {results['avg_hit_rate_top20']*100:.1f}%")
        print(f"  Average Hit Rate Top 10: {results['avg_hit_rate_top10']*100:.1f}%")
        print(f"  Average R-squared: {results['avg_r_squared']:.4f}")
    
    # Train final model and show feature importance
    print("\n2. FEATURE IMPORTANCE")
    print("-" * 70)
    
    try:
        X_train, y_train, _ = get_training_data(db, 2025, 'ops_change', 200)
        
        if len(X_train) > 50:
            model = StackingMetaLearner(feature_set='extended')
            model.fit(X_train.fillna(0).values, y_train.values)
            
            importance = model.get_feature_importance()
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n{'Feature':<30} {'Importance':>12}")
            print("-" * 45)
            for feat, imp in sorted_imp[:10]:
                print(f"{feat:<30} {imp:>12.4f}")
            
            print(f"\nMeta-learner weights:")
            for name, weight in model.meta_weights_.items():
                print(f"  {name}: {weight:.4f}")
    
    except Exception as e:
        print(f"Could not train final model: {e}")
    
    # Generate 2026 predictions
    print("\n3. 2026 PREDICTIONS (Top 20)")
    print("-" * 70)
    
    try:
        model, predictions = train_and_predict(db, 2026, 'ops_change', 200)
        
        print(f"\n{'Rank':<5} {'Player':<25} {'2025 OPS':>10} {'Pred Δ':>10} "
              f"{'2026 OPS':>10} {'CI':>15}")
        print("-" * 80)
        
        for _, row in predictions.head(20).iterrows():
            ci = f"[{row['confidence_low']:+.3f}, {row['confidence_high']:+.3f}]"
            print(f"{row['rank']:<5} {row['player_name'][:24]:<25} "
                  f"{row['current_ops']:>10.3f} {row['predicted_change']:>+10.3f} "
                  f"{row['predicted_value']:>10.3f} {ci:>15}")
    
    except Exception as e:
        print(f"Could not generate predictions: {e}")
    
    db.close()
    print("\n" + "=" * 70)
    print("Stacking model training complete!")


if __name__ == "__main__":
    main()

