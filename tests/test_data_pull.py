"""
test_data_pull.py
Tests for multi-year data pulling from Baseball Savant and FanGraphs.

Following TDD: These tests define expected behavior before implementation.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pull_multi_year import (
    pull_statcast_data,
    pull_fangraphs_batting,
    pull_all_years,
    validate_statcast_data,
    validate_fangraphs_data,
    YEARS,
    MIN_PA
)


class TestDataPullConfiguration:
    """Test configuration constants."""
    
    def test_years_range(self):
        """Verify we're pulling 2015-2025 data."""
        assert YEARS == list(range(2015, 2026))
        assert len(YEARS) == 11
    
    def test_min_pa_threshold(self):
        """Verify minimum PA threshold."""
        assert MIN_PA == 200


class TestStatcastDataValidation:
    """Tests for Statcast data validation."""
    
    def test_validate_statcast_valid_data(self):
        """Test validation passes for valid data."""
        df = pd.DataFrame({
            'player_id': [123456, 234567],
            'last_name, first_name': ['Trout, Mike', 'Ohtani, Shohei'],
            'max_hit_speed': [118.5, 119.2],
            'hard_hit_percent': [55.2, 58.1],
            'avg_hit_speed': [92.3, 94.1]
        })
        
        result = validate_statcast_data(df)
        assert result is True
    
    def test_validate_statcast_missing_columns(self):
        """Test validation fails for missing required columns."""
        df = pd.DataFrame({
            'player_id': [123456],
            'player_name': ['Trout, Mike']
            # Missing max_hit_speed, hard_hit_percent
        })
        
        result = validate_statcast_data(df)
        assert result is False
    
    def test_validate_statcast_invalid_max_ev(self):
        """Test validation catches invalid max exit velocity."""
        df = pd.DataFrame({
            'player_id': [123456],
            'last_name, first_name': ['Test, Player'],
            'max_hit_speed': [150.0],  # Impossible - too high
            'hard_hit_percent': [45.0],
            'avg_hit_speed': [90.0]
        })
        
        result = validate_statcast_data(df)
        assert result is False
    
    def test_validate_statcast_invalid_hard_hit(self):
        """Test validation catches invalid hard hit percentage."""
        df = pd.DataFrame({
            'player_id': [123456],
            'last_name, first_name': ['Test, Player'],
            'max_hit_speed': [115.0],
            'hard_hit_percent': [110.0],  # Invalid - over 100%
            'avg_hit_speed': [90.0]
        })
        
        result = validate_statcast_data(df)
        assert result is False


class TestFanGraphsDataValidation:
    """Tests for FanGraphs data validation."""
    
    def test_validate_fangraphs_valid_data(self):
        """Test validation passes for valid data."""
        df = pd.DataFrame({
            'playerid': [10155, 19755],
            'Name': ['Mike Trout', 'Shohei Ohtani'],
            'PA': [500, 600],
            'AVG': [0.280, 0.310],
            'OBP': [0.380, 0.390],
            'SLG': [0.530, 0.640],
            'wRC+': [160, 180]
        })
        
        result = validate_fangraphs_data(df)
        assert result is True
    
    def test_validate_fangraphs_missing_columns(self):
        """Test validation fails for missing columns."""
        df = pd.DataFrame({
            'playerid': [10155],
            'Name': ['Mike Trout']
            # Missing PA, AVG, OBP, SLG
        })
        
        result = validate_fangraphs_data(df)
        assert result is False
    
    def test_validate_fangraphs_low_pa(self):
        """Test validation catches players below MIN_PA."""
        df = pd.DataFrame({
            'playerid': [10155],
            'Name': ['Test Player'],
            'PA': [50],  # Below MIN_PA
            'AVG': [0.280],
            'OBP': [0.350],
            'SLG': [0.450],
            'wRC+': [120]
        })
        
        # Data should still be valid, filtering happens later
        result = validate_fangraphs_data(df)
        assert result is True


class TestDataPullFunctions:
    """Tests for data pull functions with mocked API calls."""
    
    @patch('scripts.pull_multi_year.statcast_batter_exitvelo_barrels')
    def test_pull_statcast_data_success(self, mock_statcast):
        """Test successful Statcast data pull."""
        mock_df = pd.DataFrame({
            'player_id': [123456, 234567],
            'last_name, first_name': ['Trout, Mike', 'Ohtani, Shohei'],
            'max_hit_speed': [118.5, 119.2],
            'hard_hit_percent': [55.2, 58.1],
            'avg_hit_speed': [92.3, 94.1],
            'brl_percent': [15.2, 18.5]
        })
        mock_statcast.return_value = mock_df
        
        result = pull_statcast_data(2024)
        
        assert result is not None
        assert len(result) == 2
        mock_statcast.assert_called_once()
    
    @patch('scripts.pull_multi_year.statcast_batter_exitvelo_barrels')
    def test_pull_statcast_data_api_error(self, mock_statcast):
        """Test handling of API errors."""
        mock_statcast.side_effect = Exception("API Error")
        
        result = pull_statcast_data(2024)
        
        # Should return None on error, not crash
        assert result is None
    
    @patch('scripts.pull_multi_year.batting_stats')
    def test_pull_fangraphs_data_success(self, mock_batting):
        """Test successful FanGraphs data pull."""
        mock_df = pd.DataFrame({
            'IDfg': [10155, 19755],
            'Name': ['Mike Trout', 'Shohei Ohtani'],
            'PA': [500, 600],
            'AVG': [0.280, 0.310],
            'OBP': [0.380, 0.390],
            'SLG': [0.530, 0.640],
            'wRC+': [160, 180]
        })
        mock_batting.return_value = mock_df
        
        result = pull_fangraphs_batting(2024)
        
        assert result is not None
        assert len(result) == 2
        mock_batting.assert_called_once()


class TestPullAllYears:
    """Tests for pulling all years of data."""
    
    @patch('scripts.pull_multi_year.pull_statcast_data')
    @patch('scripts.pull_multi_year.pull_fangraphs_batting')
    def test_pull_all_years_structure(self, mock_fg, mock_sc):
        """Test that pull_all_years returns proper structure."""
        mock_sc.return_value = pd.DataFrame({
            'player_id': [123456],
            'last_name, first_name': ['Test, Player'],
            'max_hit_speed': [115.0],
            'hard_hit_percent': [45.0],
            'avg_hit_speed': [90.0]
        })
        mock_fg.return_value = pd.DataFrame({
            'playerid': [10155],
            'Name': ['Test Player'],
            'PA': [500],
            'AVG': [0.280],
            'OBP': [0.350],
            'SLG': [0.450],
            'wRC+': [120]
        })
        
        result = pull_all_years([2024])
        
        assert 'statcast' in result
        assert 'fangraphs' in result
        assert 2024 in result['statcast']
        assert 2024 in result['fangraphs']


class TestDataOutput:
    """Tests for data output and storage."""
    
    def test_output_directory_structure(self):
        """Verify output directory exists or is created."""
        data_dir = Path(__file__).parent.parent / "data"
        # Directory should be creatable
        data_dir.mkdir(parents=True, exist_ok=True)
        assert data_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

