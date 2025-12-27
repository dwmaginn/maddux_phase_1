"""
test_maddux_algorithm.py
Tests for MADDUX algorithm: delta calculations and scoring.

Following TDD: These tests define expected behavior before implementation.
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase, create_schema
from scripts.maddux_algorithm import (
    calculate_deltas,
    calculate_maddux_score,
    calculate_all_deltas,
    get_consecutive_seasons,
    DEFAULT_HARD_HIT_WEIGHT,
    store_deltas_in_db,
    get_top_maddux_scores
)


@pytest.fixture
def temp_db():
    """Create a temporary database with sample data."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = MadduxDatabase(db_path)
    create_schema(db)
    
    # Insert sample players
    db.execute("INSERT INTO players (id, mlb_id, player_name) VALUES (1, 123456, 'Mike Trout')")
    db.execute("INSERT INTO players (id, mlb_id, player_name) VALUES (2, 234567, 'Shohei Ohtani')")
    db.execute("INSERT INTO players (id, mlb_id, player_name) VALUES (3, 345678, 'Aaron Judge')")
    
    # Insert Statcast seasons for player 1 (improving metrics)
    db.execute("""
        INSERT INTO statcast_seasons (player_id, year, max_ev, hard_hit_pct, avg_ev, barrel_pct)
        VALUES (1, 2023, 115.0, 50.0, 92.0, 15.0)
    """)
    db.execute("""
        INSERT INTO statcast_seasons (player_id, year, max_ev, hard_hit_pct, avg_ev, barrel_pct)
        VALUES (1, 2024, 118.0, 55.0, 94.0, 18.0)
    """)
    
    # Insert Statcast seasons for player 2 (declining metrics)
    db.execute("""
        INSERT INTO statcast_seasons (player_id, year, max_ev, hard_hit_pct, avg_ev, barrel_pct)
        VALUES (2, 2023, 120.0, 58.0, 95.0, 20.0)
    """)
    db.execute("""
        INSERT INTO statcast_seasons (player_id, year, max_ev, hard_hit_pct, avg_ev, barrel_pct)
        VALUES (2, 2024, 118.0, 55.0, 93.0, 18.0)
    """)
    
    # Insert Statcast seasons for player 3 (stable metrics)
    db.execute("""
        INSERT INTO statcast_seasons (player_id, year, max_ev, hard_hit_pct, avg_ev, barrel_pct)
        VALUES (3, 2023, 121.0, 60.0, 96.0, 25.0)
    """)
    db.execute("""
        INSERT INTO statcast_seasons (player_id, year, max_ev, hard_hit_pct, avg_ev, barrel_pct)
        VALUES (3, 2024, 121.5, 61.0, 96.5, 26.0)
    """)
    
    # Insert FanGraphs seasons for OPS tracking
    db.execute("""
        INSERT INTO fangraphs_seasons (player_id, year, pa, avg, obp, slg, ops, wrc_plus)
        VALUES (1, 2023, 500, 0.280, 0.380, 0.530, 0.910, 160)
    """)
    db.execute("""
        INSERT INTO fangraphs_seasons (player_id, year, pa, avg, obp, slg, ops, wrc_plus)
        VALUES (1, 2024, 550, 0.290, 0.400, 0.580, 0.980, 175)
    """)
    
    db.execute("""
        INSERT INTO fangraphs_seasons (player_id, year, pa, avg, obp, slg, ops, wrc_plus)
        VALUES (2, 2023, 600, 0.310, 0.390, 0.640, 1.030, 180)
    """)
    db.execute("""
        INSERT INTO fangraphs_seasons (player_id, year, pa, avg, obp, slg, ops, wrc_plus)
        VALUES (2, 2024, 650, 0.280, 0.360, 0.580, 0.940, 155)
    """)
    
    db.commit()
    
    yield db
    
    # Cleanup
    db.close()
    if os.path.exists(db_path):
        os.remove(db_path)


class TestDeltaConfiguration:
    """Tests for MADDUX algorithm configuration."""
    
    def test_default_weight(self):
        """Test default hard hit weight is 2.1."""
        assert DEFAULT_HARD_HIT_WEIGHT == 2.1


class TestDeltaCalculations:
    """Tests for year-over-year delta calculations."""
    
    def test_calculate_deltas_positive(self):
        """Test delta calculation for improving player."""
        prev_season = {'max_ev': 115.0, 'hard_hit_pct': 50.0}
        curr_season = {'max_ev': 118.0, 'hard_hit_pct': 55.0}
        
        deltas = calculate_deltas(prev_season, curr_season)
        
        assert deltas['delta_max_ev'] == 3.0
        assert deltas['delta_hard_hit_pct'] == 5.0
    
    def test_calculate_deltas_negative(self):
        """Test delta calculation for declining player."""
        prev_season = {'max_ev': 120.0, 'hard_hit_pct': 58.0}
        curr_season = {'max_ev': 118.0, 'hard_hit_pct': 55.0}
        
        deltas = calculate_deltas(prev_season, curr_season)
        
        assert deltas['delta_max_ev'] == -2.0
        assert deltas['delta_hard_hit_pct'] == -3.0
    
    def test_calculate_deltas_zero(self):
        """Test delta calculation for stable player."""
        prev_season = {'max_ev': 115.0, 'hard_hit_pct': 50.0}
        curr_season = {'max_ev': 115.0, 'hard_hit_pct': 50.0}
        
        deltas = calculate_deltas(prev_season, curr_season)
        
        assert deltas['delta_max_ev'] == 0.0
        assert deltas['delta_hard_hit_pct'] == 0.0


class TestMadduxScoreCalculation:
    """Tests for MADDUX score calculation."""
    
    def test_calculate_maddux_score_positive(self):
        """Test MADDUX score for improving player."""
        # Formula: MADDUX Score = Δ Max EV + (2.1 × Δ Hard Hit%)
        delta_max_ev = 3.0
        delta_hard_hit_pct = 5.0
        
        score = calculate_maddux_score(delta_max_ev, delta_hard_hit_pct)
        
        # 3.0 + (2.1 × 5.0) = 3.0 + 10.5 = 13.5
        assert score == pytest.approx(13.5, 0.01)
    
    def test_calculate_maddux_score_negative(self):
        """Test MADDUX score for declining player."""
        delta_max_ev = -2.0
        delta_hard_hit_pct = -3.0
        
        score = calculate_maddux_score(delta_max_ev, delta_hard_hit_pct)
        
        # -2.0 + (2.1 × -3.0) = -2.0 + -6.3 = -8.3
        assert score == pytest.approx(-8.3, 0.01)
    
    def test_calculate_maddux_score_custom_weight(self):
        """Test MADDUX score with custom weight."""
        delta_max_ev = 3.0
        delta_hard_hit_pct = 5.0
        
        score = calculate_maddux_score(delta_max_ev, delta_hard_hit_pct, weight=3.0)
        
        # 3.0 + (3.0 × 5.0) = 3.0 + 15.0 = 18.0
        assert score == pytest.approx(18.0, 0.01)
    
    def test_calculate_maddux_score_mixed(self):
        """Test MADDUX score with mixed delta directions."""
        delta_max_ev = 2.0
        delta_hard_hit_pct = -1.0
        
        score = calculate_maddux_score(delta_max_ev, delta_hard_hit_pct)
        
        # 2.0 + (2.1 × -1.0) = 2.0 - 2.1 = -0.1
        assert score == pytest.approx(-0.1, 0.01)


class TestConsecutiveSeasons:
    """Tests for finding consecutive seasons."""
    
    def test_get_consecutive_seasons(self, temp_db):
        """Test finding players with consecutive seasons."""
        pairs = get_consecutive_seasons(temp_db, year_from=2023, year_to=2024)
        
        # Should have 3 players with data in both years
        assert len(pairs) == 3
    
    def test_get_consecutive_seasons_missing_data(self, temp_db):
        """Test handling of missing consecutive data."""
        pairs = get_consecutive_seasons(temp_db, year_from=2022, year_to=2023)
        
        # No players have 2022 data
        assert len(pairs) == 0


class TestCalculateAllDeltas:
    """Tests for batch delta calculation."""
    
    def test_calculate_all_deltas(self, temp_db):
        """Test calculating deltas for all players."""
        results = calculate_all_deltas(temp_db, year_from=2023, year_to=2024)
        
        assert len(results) == 3
        
        # Check that results have required fields
        for result in results:
            assert 'player_id' in result
            assert 'delta_max_ev' in result
            assert 'delta_hard_hit_pct' in result
            assert 'maddux_score' in result


class TestDatabaseStorage:
    """Tests for storing deltas in database."""
    
    def test_store_deltas_in_db(self, temp_db):
        """Test storing calculated deltas in database."""
        deltas = [
            {
                'player_id': 1,
                'year_from': 2023,
                'year_to': 2024,
                'delta_max_ev': 3.0,
                'delta_hard_hit_pct': 5.0,
                'maddux_score': 13.5,
                'delta_ops': 0.07
            }
        ]
        
        count = store_deltas_in_db(temp_db, deltas)
        
        assert count == 1
        
        # Verify data was stored
        result = temp_db.query("SELECT COUNT(*) FROM player_deltas")
        assert result[0][0] == 1


class TestTopScores:
    """Tests for getting top MADDUX scores."""
    
    def test_get_top_maddux_scores(self, temp_db):
        """Test getting top scoring players."""
        # First calculate and store deltas
        results = calculate_all_deltas(temp_db, year_from=2023, year_to=2024)
        store_deltas_in_db(temp_db, results)
        
        # Get top scores
        top = get_top_maddux_scores(temp_db, year_from=2023, year_to=2024, limit=10)
        
        assert len(top) <= 10
        
        # Should be sorted by MADDUX score descending
        if len(top) > 1:
            assert top[0]['maddux_score'] >= top[1]['maddux_score']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

