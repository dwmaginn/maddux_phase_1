"""
test_stacking_model.py
Tests for the Stacking Meta-Learner model.

Tests cover:
- Model initialization and configuration
- Training and fitting
- Prediction with confidence intervals
- Model evaluation metrics
- Feature importance extraction
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if sklearn is available
try:
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import stacking model if sklearn available
if SKLEARN_AVAILABLE:
    from scripts.stacking_model import (
        StackingMetaLearner,
        ModelMetrics,
        PredictionResult,
        BASE_FEATURES,
        EXTENDED_FEATURES,
    )


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 200
    
    # Generate features
    data = {
        'underperformance_gap': np.random.normal(0, 30, n_samples),
        'xslg_gap_scaled': np.random.normal(0, 40, n_samples),
        'deviation_from_baseline': np.random.normal(0, 0.030, n_samples),
        'career_peak_deviation': np.random.normal(0.050, 0.030, n_samples),
        'improvement_momentum': np.random.normal(0, 0.020, n_samples),
        'combined_luck_index': np.random.normal(0, 25, n_samples),
        'quality_zscore': np.random.normal(0, 1, n_samples),
        'hard_hit_zscore': np.random.normal(0, 1, n_samples),
        'age_factor': np.random.normal(0, 0.05, n_samples),
    }
    
    # Generate target with some correlation to features
    y = (0.002 * data['underperformance_gap'] +
         0.5 * data['deviation_from_baseline'] +
         -0.3 * data['improvement_momentum'] +  # Negative predictor!
         np.random.normal(0, 0.030, n_samples))
    
    X = pd.DataFrame(data)
    
    return X, y


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestStackingModelInit:
    """Tests for model initialization."""
    
    def test_model_init_default(self):
        """Test model initializes with default parameters."""
        model = StackingMetaLearner()
        
        assert model is not None
        assert not model.is_fitted
        assert 'ridge' in model.base_models
        assert 'lasso' in model.base_models
        assert 'gbm' in model.base_models
    
    def test_model_init_base_features(self):
        """Test model with base feature set."""
        model = StackingMetaLearner(feature_set='base')
        
        assert model.feature_names == BASE_FEATURES
    
    def test_model_init_extended_features(self):
        """Test model with extended feature set."""
        model = StackingMetaLearner(feature_set='extended')
        
        assert model.feature_names == EXTENDED_FEATURES
    
    def test_model_init_custom_params(self):
        """Test model with custom hyperparameters."""
        model = StackingMetaLearner(
            ridge_alpha=2.0,
            lasso_alpha=0.05,
            gbm_n_estimators=50,
            meta_alpha=0.2
        )
        
        assert model.base_models['ridge'].alpha == 2.0


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestStackingModelFit:
    """Tests for model fitting."""
    
    def test_model_fit(self, sample_data):
        """Test basic model fitting."""
        X, y = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        model.fit(X[model.feature_names].fillna(0).values, y)
        
        assert model.is_fitted
    
    def test_model_fit_stores_importance(self, sample_data):
        """Test that fitting stores feature importance."""
        X, y = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        model.fit(X[model.feature_names].fillna(0).values, y)
        
        importance = model.get_feature_importance()
        
        assert len(importance) > 0
        assert sum(importance.values()) == pytest.approx(1.0, 0.01)
    
    def test_model_fit_stores_meta_weights(self, sample_data):
        """Test that fitting stores meta-learner weights."""
        X, y = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        model.fit(X[model.feature_names].fillna(0).values, y)
        
        assert 'ridge' in model.meta_weights_
        assert 'lasso' in model.meta_weights_
        assert 'gbm' in model.meta_weights_


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestStackingModelPredict:
    """Tests for model predictions."""
    
    def test_model_predict_basic(self, sample_data):
        """Test basic prediction."""
        X, y = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        X_arr = X[model.feature_names].fillna(0).values
        model.fit(X_arr, y)
        
        predictions = model.predict(X_arr[:10])
        
        assert len(predictions) == 10
        assert all(np.isfinite(predictions))
    
    def test_model_predict_not_fitted_raises(self, sample_data):
        """Test that predicting without fitting raises error."""
        X, _ = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X[model.feature_names].fillna(0).values[:10])
    
    def test_model_predict_with_confidence(self, sample_data):
        """Test prediction with confidence intervals."""
        X, y = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        X_arr = X[model.feature_names].fillna(0).values
        model.fit(X_arr, y)
        
        predictions, lower, upper = model.predict_with_confidence(
            X_arr[:10], n_bootstrap=50
        )
        
        assert len(predictions) == 10
        assert len(lower) == 10
        assert len(upper) == 10
        
        # Confidence intervals should bracket predictions
        assert all(lower <= predictions)
        assert all(predictions <= upper)
    
    def test_confidence_interval_width(self, sample_data):
        """Test that confidence intervals have reasonable width."""
        X, y = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        X_arr = X[model.feature_names].fillna(0).values
        model.fit(X_arr, y)
        
        _, lower, upper = model.predict_with_confidence(
            X_arr[:10], n_bootstrap=100, confidence=0.95
        )
        
        # CI width should be positive but not too wide
        widths = upper - lower
        assert all(widths >= 0)
        assert all(widths < 0.5)  # Reasonable for OPS change


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestStackingModelEvaluate:
    """Tests for model evaluation."""
    
    def test_model_evaluate_metrics(self, sample_data):
        """Test evaluation returns correct metrics."""
        X, y = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        X_arr = X[model.feature_names].fillna(0).values
        model.fit(X_arr, y)
        
        metrics = model.evaluate(X_arr, y)
        
        assert isinstance(metrics, ModelMetrics)
        assert hasattr(metrics, 'correlation')
        assert hasattr(metrics, 'r_squared')
        assert hasattr(metrics, 'mae')
        assert hasattr(metrics, 'hit_rate_top20')
    
    def test_model_evaluate_correlation_range(self, sample_data):
        """Test that correlation is in valid range."""
        X, y = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        X_arr = X[model.feature_names].fillna(0).values
        model.fit(X_arr, y)
        
        metrics = model.evaluate(X_arr, y)
        
        # Correlation should be between -1 and 1
        assert -1 <= metrics.correlation <= 1
    
    def test_model_evaluate_hit_rate_range(self, sample_data):
        """Test that hit rates are in valid range."""
        X, y = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        X_arr = X[model.feature_names].fillna(0).values
        model.fit(X_arr, y)
        
        metrics = model.evaluate(X_arr, y)
        
        # Hit rates should be between 0 and 1
        assert 0 <= metrics.hit_rate_top20 <= 1
        assert 0 <= metrics.hit_rate_top10 <= 1


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestFeatureImportance:
    """Tests for feature importance extraction."""
    
    def test_feature_importance_sums_to_one(self, sample_data):
        """Test that importance values sum to 1."""
        X, y = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        model.fit(X[model.feature_names].fillna(0).values, y)
        
        importance = model.get_feature_importance()
        
        assert sum(importance.values()) == pytest.approx(1.0, 0.01)
    
    def test_feature_importance_all_positive(self, sample_data):
        """Test that importance values are non-negative."""
        X, y = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        model.fit(X[model.feature_names].fillna(0).values, y)
        
        importance = model.get_feature_importance()
        
        assert all(v >= 0 for v in importance.values())
    
    def test_feature_importance_matches_features(self, sample_data):
        """Test that importance keys match used features."""
        X, y = sample_data
        
        model = StackingMetaLearner(feature_set='base')
        model.fit(X[model.feature_names].fillna(0).values, y)
        
        importance = model.get_feature_importance()
        
        # All importance keys should be in available features
        for key in importance.keys():
            assert key in model.available_features_


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestModelReproducibility:
    """Tests for model reproducibility."""
    
    def test_prediction_reproducibility(self, sample_data):
        """Test that predictions are reproducible with same data."""
        X, y = sample_data
        
        # Train two models with same data
        model1 = StackingMetaLearner(feature_set='base')
        model2 = StackingMetaLearner(feature_set='base')
        
        X_arr = X[model1.feature_names].fillna(0).values
        
        model1.fit(X_arr, y)
        model2.fit(X_arr, y)
        
        pred1 = model1.predict(X_arr[:10])
        pred2 = model2.predict(X_arr[:10])
        
        # Predictions should be close (GBM has randomness)
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=2)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestModelConstants:
    """Tests for model constants and configurations."""
    
    def test_base_features_defined(self):
        """Test that BASE_FEATURES is properly defined."""
        assert len(BASE_FEATURES) >= 5
        assert 'underperformance_gap' in BASE_FEATURES
        assert 'deviation_from_baseline' in BASE_FEATURES
    
    def test_extended_features_includes_base(self):
        """Test that EXTENDED_FEATURES includes all BASE_FEATURES."""
        for feat in BASE_FEATURES:
            assert feat in EXTENDED_FEATURES
    
    def test_extended_features_has_extras(self):
        """Test that EXTENDED_FEATURES has additional features."""
        assert len(EXTENDED_FEATURES) > len(BASE_FEATURES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


