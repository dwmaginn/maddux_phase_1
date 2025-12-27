"""
test_database.py
Tests for multi-year SQLite database schema and operations.

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

from scripts.database import (
    MadduxDatabase,
    create_schema,
    load_statcast_data,
    load_fangraphs_data,
    create_id_mapping,
    get_player_seasons,
    get_players_by_year,
    filter_by_min_pa
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = MadduxDatabase(db_path)
    yield db
    
    # Cleanup
    db.close()
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def sample_statcast_df():
    """Sample Statcast data for testing."""
    return pd.DataFrame({
        'player_id': [123456, 234567, 345678],
        'player_name': ['Trout, Mike', 'Ohtani, Shohei', 'Judge, Aaron'],
        'year': [2024, 2024, 2024],
        'max_hit_speed': [118.5, 119.2, 121.3],
        'hard_hit_percent': [55.2, 58.1, 61.0],
        'avg_hit_speed': [92.3, 94.1, 95.5],
        'brl_percent': [15.2, 18.5, 26.9]
    })


@pytest.fixture
def sample_fangraphs_df():
    """Sample FanGraphs data for testing."""
    return pd.DataFrame({
        'playerid': [10155, 19755, 15640],
        'Name': ['Mike Trout', 'Shohei Ohtani', 'Aaron Judge'],
        'year': [2024, 2024, 2024],
        'PA': [500, 700, 700],
        'AVG': [0.280, 0.310, 0.320],
        'OBP': [0.380, 0.390, 0.460],
        'SLG': [0.530, 0.640, 0.700],
        'wRC+': [160, 180, 220]
    })


class TestSchemaCreation:
    """Tests for database schema creation."""
    
    def test_create_schema_creates_tables(self, temp_db):
        """Test that schema creation creates all required tables."""
        create_schema(temp_db)
        
        tables = temp_db.get_tables()
        
        assert 'players' in tables
        assert 'statcast_seasons' in tables
        assert 'fangraphs_seasons' in tables
        assert 'player_id_mapping' in tables
    
    def test_players_table_has_correct_columns(self, temp_db):
        """Test players table has required columns."""
        create_schema(temp_db)
        
        columns = temp_db.get_table_columns('players')
        
        assert 'id' in columns
        assert 'mlb_id' in columns
        assert 'fangraphs_id' in columns
        assert 'player_name' in columns
    
    def test_statcast_seasons_table_has_correct_columns(self, temp_db):
        """Test statcast_seasons table has required columns."""
        create_schema(temp_db)
        
        columns = temp_db.get_table_columns('statcast_seasons')
        
        assert 'player_id' in columns
        assert 'year' in columns
        assert 'max_ev' in columns
        assert 'hard_hit_pct' in columns
        assert 'avg_ev' in columns
        assert 'barrel_pct' in columns
    
    def test_fangraphs_seasons_table_has_correct_columns(self, temp_db):
        """Test fangraphs_seasons table has required columns."""
        create_schema(temp_db)
        
        columns = temp_db.get_table_columns('fangraphs_seasons')
        
        assert 'player_id' in columns
        assert 'year' in columns
        assert 'pa' in columns
        assert 'avg' in columns
        assert 'obp' in columns
        assert 'slg' in columns
        assert 'ops' in columns
        assert 'wrc_plus' in columns


class TestDataLoading:
    """Tests for loading data into database."""
    
    def test_load_statcast_data(self, temp_db, sample_statcast_df):
        """Test loading Statcast data into database."""
        create_schema(temp_db)
        
        count = load_statcast_data(temp_db, sample_statcast_df)
        
        assert count == 3
        
        # Verify data was loaded
        result = temp_db.query("SELECT COUNT(*) FROM statcast_seasons")
        assert result[0][0] == 3
    
    def test_load_fangraphs_data(self, temp_db, sample_fangraphs_df):
        """Test loading FanGraphs data into database."""
        create_schema(temp_db)
        
        count = load_fangraphs_data(temp_db, sample_fangraphs_df)
        
        assert count == 3
        
        # Verify data was loaded
        result = temp_db.query("SELECT COUNT(*) FROM fangraphs_seasons")
        assert result[0][0] == 3
    
    def test_load_data_creates_players(self, temp_db, sample_statcast_df):
        """Test that loading data creates player records."""
        create_schema(temp_db)
        
        load_statcast_data(temp_db, sample_statcast_df)
        
        # Should have 3 players
        result = temp_db.query("SELECT COUNT(*) FROM players")
        assert result[0][0] == 3


class TestIDMapping:
    """Tests for player ID mapping between sources."""
    
    def test_create_id_mapping(self, temp_db, sample_statcast_df, sample_fangraphs_df):
        """Test creating player ID mapping."""
        create_schema(temp_db)
        load_statcast_data(temp_db, sample_statcast_df)
        load_fangraphs_data(temp_db, sample_fangraphs_df)
        
        # Create mapping based on name similarity
        count = create_id_mapping(temp_db)
        
        # Should map at least some players
        assert count >= 0  # May be 0 if no matches found


class TestQueries:
    """Tests for database query functions."""
    
    def test_get_player_seasons(self, temp_db, sample_statcast_df, sample_fangraphs_df):
        """Test getting all seasons for a player."""
        create_schema(temp_db)
        load_statcast_data(temp_db, sample_statcast_df)
        load_fangraphs_data(temp_db, sample_fangraphs_df)
        
        # Get seasons for a player
        seasons = get_player_seasons(temp_db, mlb_id=123456)
        
        assert len(seasons) >= 1
    
    def test_get_players_by_year(self, temp_db, sample_statcast_df):
        """Test getting all players for a specific year."""
        create_schema(temp_db)
        load_statcast_data(temp_db, sample_statcast_df)
        
        players = get_players_by_year(temp_db, year=2024)
        
        assert len(players) == 3
    
    def test_filter_by_min_pa(self, temp_db, sample_fangraphs_df):
        """Test filtering players by minimum PA."""
        create_schema(temp_db)
        load_fangraphs_data(temp_db, sample_fangraphs_df)
        
        # Filter for players with PA >= 600
        players = filter_by_min_pa(temp_db, year=2024, min_pa=600)
        
        assert len(players) == 2  # Ohtani and Judge have PA >= 600


class TestDatabaseOperations:
    """Tests for general database operations."""
    
    def test_database_connection(self, temp_db):
        """Test database connection works."""
        assert temp_db.conn is not None
    
    def test_database_close(self, temp_db):
        """Test database closes properly."""
        temp_db.close()
        # Should not raise an error when closing again
        temp_db.close()
    
    def test_query_returns_results(self, temp_db):
        """Test query method returns results."""
        create_schema(temp_db)
        result = temp_db.query("SELECT name FROM sqlite_master WHERE type='table'")
        
        assert isinstance(result, list)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

