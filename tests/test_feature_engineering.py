"""
test_feature_engineering.py
Tests for feature engineering module.

Tests cover:
- Luck/underperformance features (xwOBA gap, BABIP gap)
- Baseline/trend features (deviation from baseline, momentum)
- Bat quality features (bat speed z-score, whiff-adjusted power)
- Feature correlation analysis
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase, create_schema
from scripts.feature_engineering import (
    calculate_underperformance_gap,
    calculate_xslg_gap,
    calculate_babip_gap,
    calculate_iso_gap,
    calculate_combined_luck_index,
    calculate_age_factor,
    calculate_bat_speed_zscore,
    calculate_swing_length_factor,
    calculate_squared_up_quality,
    calculate_whiff_adjusted_power,
    calculate_original_maddux_score,
)


class TestLuckFeatures:
    """Tests for luck/underperformance feature calculations."""
    
    def test_underperformance_gap_positive(self):
        """Test xwOBA > wOBA indicates underperformance (unlucky)."""
        xwoba = 0.350
        woba = 0.320
        
        gap = calculate_underperformance_gap(xwoba, woba)
        
        # (0.350 - 0.320) * 1000 = 30
        assert gap == pytest.approx(30.0, 0.1)
    
    def test_underperformance_gap_negative(self):
        """Test xwOBA < wOBA indicates overperformance (lucky)."""
        xwoba = 0.320
        woba = 0.350
        
        gap = calculate_underperformance_gap(xwoba, woba)
        
        # (0.320 - 0.350) * 1000 = -30
        assert gap == pytest.approx(-30.0, 0.1)
    
    def test_underperformance_gap_none(self):
        """Test handling of None values."""
        assert calculate_underperformance_gap(None, 0.320) is None
        assert calculate_underperformance_gap(0.350, None) is None
        assert calculate_underperformance_gap(None, None) is None
    
    def test_xslg_gap_calculation(self):
        """Test xSLG gap calculation."""
        xslg = 0.500
        slg = 0.450
        
        gap = calculate_xslg_gap(xslg, slg)
        
        assert gap == pytest.approx(50.0, 0.1)
    
    def test_babip_gap_calculation(self):
        """Test BABIP gap calculation."""
        xba = 0.280
        babip = 0.300
        
        gap = calculate_babip_gap(xba, babip)
        
        # (0.280 - 0.300) * 1000 = -20 (overperforming on BABIP)
        assert gap == pytest.approx(-20.0, 0.1)
    
    def test_iso_gap_calculation(self):
        """Test ISO gap calculation."""
        xslg = 0.500
        xba = 0.280
        iso = 0.200  # Expected ISO = xSLG - xBA = 0.220
        
        gap = calculate_iso_gap(xslg, xba, iso)
        
        # (0.220 - 0.200) * 1000 = 20
        assert gap == pytest.approx(20.0, 0.1)
    
    def test_combined_luck_index(self):
        """Test combined luck index calculation."""
        underperformance_gap = 30.0
        xslg_gap = 50.0
        babip_gap = -20.0
        
        index = calculate_combined_luck_index(
            underperformance_gap, xslg_gap, babip_gap
        )
        
        # Weighted: 30*0.5 + 50*0.3 + (-20)*0.2 = 15 + 15 - 4 = 26
        assert index is not None
        assert index > 0  # Overall unlucky
    
    def test_combined_luck_index_partial(self):
        """Test combined luck with partial data."""
        index = calculate_combined_luck_index(30.0, None, None)
        
        assert index is not None
        assert index == pytest.approx(30.0, 0.1)  # Just the xwOBA gap


class TestBaselineFeatures:
    """Tests for baseline/trend feature calculations."""
    
    def test_age_factor_young(self):
        """Test age factor for young player (pre-peak)."""
        factor = calculate_age_factor(24)
        
        # Under 26, should be positive (room to grow)
        assert factor > 0
    
    def test_age_factor_peak(self):
        """Test age factor for peak-age player."""
        factor = calculate_age_factor(27)
        
        # 26-29 is peak, should be neutral
        assert factor == 0.0
    
    def test_age_factor_old(self):
        """Test age factor for older player (post-peak)."""
        factor = calculate_age_factor(33)
        
        # Over 29, should be negative (decline risk)
        assert factor < 0
    
    def test_age_factor_none(self):
        """Test age factor with None."""
        factor = calculate_age_factor(None)
        
        assert factor == 0.0


class TestBatQualityFeatures:
    """Tests for bat quality feature calculations."""
    
    def test_bat_speed_zscore_above_average(self):
        """Test z-score for above-average bat speed."""
        avg_bat_speed = 75.0
        league_mean = 71.0
        league_std = 3.0
        
        z = calculate_bat_speed_zscore(avg_bat_speed, league_mean, league_std)
        
        # (75 - 71) / 3 = 1.33
        assert z == pytest.approx(1.33, 0.01)
    
    def test_bat_speed_zscore_below_average(self):
        """Test z-score for below-average bat speed."""
        avg_bat_speed = 68.0
        league_mean = 71.0
        league_std = 3.0
        
        z = calculate_bat_speed_zscore(avg_bat_speed, league_mean, league_std)
        
        # (68 - 71) / 3 = -1.0
        assert z == pytest.approx(-1.0, 0.01)
    
    def test_swing_length_factor(self):
        """Test swing length relative to league average."""
        swing_length = 7.8
        league_mean = 7.5
        
        factor = calculate_swing_length_factor(swing_length, league_mean)
        
        # 7.8 / 7.5 = 1.04
        assert factor == pytest.approx(1.04, 0.01)
    
    def test_squared_up_quality(self):
        """Test squared-up contact quality calculation."""
        squared_up_rate = 0.25
        avg_bat_speed = 74.0
        
        quality = calculate_squared_up_quality(squared_up_rate, avg_bat_speed)
        
        # 0.25 * (74 / 70) = 0.264
        assert quality == pytest.approx(0.264, 0.01)
    
    def test_whiff_adjusted_power(self):
        """Test whiff-adjusted power calculation."""
        barrel_pct = 15.0
        whiff_rate = 0.25
        
        power = calculate_whiff_adjusted_power(barrel_pct, whiff_rate)
        
        # 15 / 0.25 = 60
        assert power == pytest.approx(60.0, 0.1)
    
    def test_whiff_adjusted_power_low_whiff(self):
        """Test whiff-adjusted power with very low whiff (efficient hitter)."""
        barrel_pct = 10.0
        whiff_rate = 0.15
        
        power = calculate_whiff_adjusted_power(barrel_pct, whiff_rate)
        
        # 10 / 0.15 = 66.67 (efficient)
        assert power == pytest.approx(66.67, 0.1)
    
    def test_whiff_adjusted_power_high_whiff(self):
        """Test whiff-adjusted power with high whiff (volatile hitter)."""
        barrel_pct = 15.0
        whiff_rate = 0.35
        
        power = calculate_whiff_adjusted_power(barrel_pct, whiff_rate)
        
        # 15 / 0.35 = 42.86 (volatile)
        assert power == pytest.approx(42.86, 0.1)


class TestOriginalMadduxScore:
    """Tests for original MADDUX score calculation (for comparison)."""
    
    def test_original_maddux_score_positive(self):
        """Test original MADDUX score with positive deltas."""
        delta_max_ev = 3.0
        delta_hard_hit_pct = 5.0
        
        score = calculate_original_maddux_score(delta_max_ev, delta_hard_hit_pct)
        
        # 3.0 + (2.1 * 5.0) = 13.5
        assert score == pytest.approx(13.5, 0.01)
    
    def test_original_maddux_score_negative(self):
        """Test original MADDUX score with negative deltas."""
        delta_max_ev = -2.0
        delta_hard_hit_pct = -3.0
        
        score = calculate_original_maddux_score(delta_max_ev, delta_hard_hit_pct)
        
        # -2.0 + (2.1 * -3.0) = -8.3
        assert score == pytest.approx(-8.3, 0.01)


class TestFeatureEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_all_functions_handle_none(self):
        """Test that all feature functions handle None gracefully."""
        assert calculate_underperformance_gap(None, None) is None
        assert calculate_xslg_gap(None, None) is None
        assert calculate_babip_gap(None, None) is None
        assert calculate_bat_speed_zscore(None, 71.0, 3.0) is None
        assert calculate_swing_length_factor(None, 7.5) is None
        assert calculate_squared_up_quality(None, 74.0) is None
        assert calculate_whiff_adjusted_power(None, 0.25) is None
    
    def test_zero_division_handling(self):
        """Test handling of zero division cases."""
        # Whiff rate of 0 should return high value (capped)
        power = calculate_whiff_adjusted_power(15.0, 0.0)
        assert power == pytest.approx(150.0, 0.1)  # barrel_pct * 10
        
        # League std of 0 should return 0
        z = calculate_bat_speed_zscore(75.0, 71.0, 0.0)
        assert z == 0.0
        
        # League mean of 0 should return None
        factor = calculate_swing_length_factor(7.5, 0.0)
        assert factor is None


class TestFeatureIntegration:
    """Integration tests for feature calculations."""
    
    def test_feature_calculation_consistency(self):
        """Test that features produce consistent results."""
        # Run same calculation multiple times
        results = []
        for _ in range(10):
            gap = calculate_underperformance_gap(0.350, 0.320)
            results.append(gap)
        
        # All results should be identical
        assert all(r == results[0] for r in results)
    
    def test_feature_ranges(self):
        """Test that features produce values in expected ranges."""
        # Z-scores should typically be between -4 and 4
        z = calculate_bat_speed_zscore(75.0, 71.0, 3.0)
        assert -4 <= z <= 4
        
        # Swing length factor should be around 1.0
        factor = calculate_swing_length_factor(7.8, 7.5)
        assert 0.5 <= factor <= 1.5
        
        # Age factor should be small
        age_f = calculate_age_factor(27)
        assert -0.2 <= age_f <= 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


