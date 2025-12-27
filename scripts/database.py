"""
database.py
SQLite database for MADDUX Phase 1 multi-year hitter data.

Enhanced Schema supports:
- Players with MLB and FanGraphs IDs
- Multi-year Statcast data (exit velocity, hard hit %, barrels, bat tracking, expected stats)
- Multi-year FanGraphs data (PA, AVG, OBP, SLG, OPS, wRC+, wOBA, BABIP, discipline)
- Player ID mapping between sources
- Computed player features for ML models
- Flexible filtering (min PA, year ranges)

New in v2:
- Bat tracking metrics (bat speed, swing length, whiff rate, squared up)
- Expected stats (xwOBA, xBA, xSLG)
- Luck indicators (BABIP, contact%, chase rate)
- Player features table for engineered features
"""

import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd

# Try to import pybaseball for ID mapping
try:
    from pybaseball import playerid_reverse_lookup
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "database" / "maddux.db"


class MadduxDatabase:
    """
    SQLite database wrapper for MADDUX Phase 1 data.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = str(DEFAULT_DB_PATH)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def close(self):
        """Close database connection."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
    
    def execute(self, sql: str, params: tuple = None) -> sqlite3.Cursor:
        """Execute SQL statement."""
        if params:
            return self.conn.execute(sql, params)
        return self.conn.execute(sql)
    
    def executemany(self, sql: str, params_list: List[tuple]) -> sqlite3.Cursor:
        """Execute SQL statement for multiple parameter sets."""
        return self.conn.executemany(sql, params_list)
    
    def commit(self):
        """Commit transaction."""
        self.conn.commit()
    
    def query(self, sql: str, params: tuple = None) -> List[tuple]:
        """Execute query and return results as list of tuples."""
        cursor = self.execute(sql, params)
        return cursor.fetchall()
    
    def query_df(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame."""
        return pd.read_sql_query(sql, self.conn, params=params)
    
    def get_tables(self) -> List[str]:
        """Get list of all tables in database."""
        result = self.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row[0] for row in result]
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """Get list of column names for a table."""
        result = self.query(f"PRAGMA table_info({table_name})")
        return [row[1] for row in result]


def create_schema(db: MadduxDatabase) -> None:
    """
    Create database schema for MADDUX Phase 1.
    
    Tables:
        - players: Master player table with ID mappings
        - statcast_seasons: Yearly Statcast data
        - fangraphs_seasons: Yearly FanGraphs data
        - player_id_mapping: Maps MLB IDs to FanGraphs IDs
        - player_deltas: Year-over-year metric changes
        - maddux_scores: Calculated MADDUX scores
    """
    
    # Players table
    db.execute("""
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mlb_id INTEGER UNIQUE,
            fangraphs_id INTEGER,
            player_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index on mlb_id and fangraphs_id
    db.execute("CREATE INDEX IF NOT EXISTS idx_players_mlb_id ON players(mlb_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_players_fg_id ON players(fangraphs_id)")
    
    # Statcast seasons table - Enhanced with bat tracking, expected stats, and launch angle
    db.execute("""
        CREATE TABLE IF NOT EXISTS statcast_seasons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            year INTEGER NOT NULL,
            -- Exit velocity metrics
            max_ev REAL,
            avg_ev REAL,
            hard_hit_pct REAL,
            barrel_pct REAL,
            batted_balls INTEGER,
            -- Bat tracking metrics
            avg_bat_speed REAL,
            swing_length REAL,
            hard_swing_rate REAL,
            squared_up_rate REAL,
            whiff_rate REAL,
            blast_rate REAL,
            -- Launch angle metrics (new)
            avg_launch_angle REAL,
            sweet_spot_pct REAL,
            -- Expected stats
            xwoba REAL,
            xba REAL,
            xslg REAL,
            xwoba_gap REAL,
            xslg_gap REAL,
            FOREIGN KEY (player_id) REFERENCES players(id),
            UNIQUE(player_id, year)
        )
    """)
    
    db.execute("CREATE INDEX IF NOT EXISTS idx_statcast_year ON statcast_seasons(year)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_statcast_player ON statcast_seasons(player_id)")
    
    # FanGraphs seasons table - Enhanced with wOBA, BABIP, discipline metrics
    db.execute("""
        CREATE TABLE IF NOT EXISTS fangraphs_seasons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            year INTEGER NOT NULL,
            age INTEGER,
            pa INTEGER,
            avg REAL,
            obp REAL,
            slg REAL,
            ops REAL,
            iso REAL,
            wrc_plus REAL,
            -- Discipline metrics
            bb_pct REAL,
            k_pct REAL,
            contact_pct REAL,
            chase_rate REAL,
            -- Zone swing metrics (new)
            z_swing_pct REAL,
            z_contact_pct REAL,
            -- Advanced metrics
            woba REAL,
            babip REAL,
            war REAL,
            FOREIGN KEY (player_id) REFERENCES players(id),
            UNIQUE(player_id, year)
        )
    """)
    
    db.execute("CREATE INDEX IF NOT EXISTS idx_fangraphs_year ON fangraphs_seasons(year)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_fangraphs_player ON fangraphs_seasons(player_id)")
    
    # Player ID mapping table
    db.execute("""
        CREATE TABLE IF NOT EXISTS player_id_mapping (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mlb_id INTEGER UNIQUE,
            fangraphs_id INTEGER,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Player deltas table (for year-over-year changes)
    db.execute("""
        CREATE TABLE IF NOT EXISTS player_deltas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            year_from INTEGER NOT NULL,
            year_to INTEGER NOT NULL,
            delta_max_ev REAL,
            delta_hard_hit_pct REAL,
            delta_ops REAL,
            maddux_score REAL,
            FOREIGN KEY (player_id) REFERENCES players(id),
            UNIQUE(player_id, year_from, year_to)
        )
    """)
    
    db.execute("CREATE INDEX IF NOT EXISTS idx_deltas_years ON player_deltas(year_from, year_to)")
    
    # MADDUX rankings table
    db.execute("""
        CREATE TABLE IF NOT EXISTS maddux_rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            rank INTEGER,
            maddux_score REAL,
            predicted_ops_change REAL,
            actual_ops_change REAL,
            hit_rate REAL,
            FOREIGN KEY (player_id) REFERENCES players(id),
            UNIQUE(year, player_id)
        )
    """)
    
    # Player features table - Engineered features for ML models (new)
    db.execute("""
        CREATE TABLE IF NOT EXISTS player_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            year INTEGER NOT NULL,
            -- Luck/underperformance features
            underperformance_gap REAL,
            xslg_gap_scaled REAL,
            babip_gap REAL,
            iso_gap REAL,
            combined_luck_index REAL,
            -- Baseline/trend features
            deviation_from_baseline REAL,
            career_peak_deviation REAL,
            improvement_momentum REAL,
            ops_trend_2yr REAL,
            years_from_peak INTEGER,
            -- Bat quality features
            bat_speed_zscore REAL,
            swing_length_factor REAL,
            squared_up_quality REAL,
            whiff_adjusted_power REAL,
            quality_zscore REAL,
            hard_hit_zscore REAL,
            -- Age/context features
            age_factor REAL,
            pa_weight REAL,
            -- Original MADDUX score (for comparison)
            original_maddux_score REAL,
            FOREIGN KEY (player_id) REFERENCES players(id),
            UNIQUE(player_id, year)
        )
    """)
    
    db.execute("CREATE INDEX IF NOT EXISTS idx_features_year ON player_features(year)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_features_player ON player_features(player_id)")
    
    # Model predictions table - Store predictions with confidence intervals (new)
    db.execute("""
        CREATE TABLE IF NOT EXISTS model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            prediction_year INTEGER NOT NULL,
            target_year INTEGER NOT NULL,
            model_type TEXT NOT NULL,
            target_variable TEXT NOT NULL,
            predicted_value REAL,
            confidence_low REAL,
            confidence_high REAL,
            actual_value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES players(id),
            UNIQUE(player_id, prediction_year, target_year, model_type, target_variable)
        )
    """)
    
    db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_year ON model_predictions(target_year)")
    
    db.commit()
    print("Database schema created successfully (enhanced v2)")


def get_or_create_player(db: MadduxDatabase, mlb_id: int, player_name: str) -> int:
    """
    Get existing player ID or create new player record.
    
    Args:
        db: Database connection
        mlb_id: MLB player ID
        player_name: Player name
        
    Returns:
        Player ID (internal)
    """
    # Check if player exists
    result = db.query(
        "SELECT id FROM players WHERE mlb_id = ?",
        (mlb_id,)
    )
    
    if result:
        return result[0][0]
    
    # Create new player
    cursor = db.execute(
        "INSERT INTO players (mlb_id, player_name) VALUES (?, ?)",
        (mlb_id, player_name)
    )
    db.commit()
    
    return cursor.lastrowid


def load_statcast_data(db: MadduxDatabase, df: pd.DataFrame) -> int:
    """
    Load Statcast data into database (enhanced with bat tracking and expected stats).
    
    Args:
        db: Database connection
        df: DataFrame with Statcast data
        
    Returns:
        Number of rows loaded
    """
    count = 0
    for _, row in df.iterrows():
        try:
            # Get MLB ID and player name
            mlb_id = int(row.get('player_id', 0))
            player_name = row.get('player_name', row.get('last_name, first_name', 'Unknown'))
            
            if mlb_id == 0:
                continue
            
            # Get or create player
            player_id = get_or_create_player(db, mlb_id, player_name)
            
            # Get year
            year = int(row.get('year', 0))
            if year == 0:
                continue
            
            # Get exit velocity stats
            max_ev = row.get('max_hit_speed', row.get('max_ev'))
            avg_ev = row.get('avg_hit_speed', row.get('avg_ev'))
            hard_hit_pct = row.get('hard_hit_percent', row.get('ev95percent', row.get('hard_hit_pct')))
            barrel_pct = row.get('brl_percent', row.get('barrel_batted_rate', row.get('barrel_pct')))
            batted_balls = row.get('batted_balls', row.get('batted_ball_events'))
            
            # Get bat tracking stats
            avg_bat_speed = row.get('avg_bat_speed')
            swing_length = row.get('swing_length')
            hard_swing_rate = row.get('hard_swing_rate')
            squared_up_rate = row.get('squared_up_per_swing', row.get('squared_up_rate'))
            whiff_rate = row.get('whiff_per_swing', row.get('whiff_rate'))
            blast_rate = row.get('blast_per_swing', row.get('blast_rate'))
            
            # Get launch angle stats (new)
            avg_launch_angle = row.get('avg_hit_angle', row.get('avg_launch_angle'))
            sweet_spot_pct = row.get('anglesweetspotpercent', row.get('sweet_spot_pct'))
            
            # Get expected stats
            xwoba = row.get('xwoba', row.get('est_woba'))
            xba = row.get('xba', row.get('est_ba'))
            xslg = row.get('xslg', row.get('est_slg'))
            
            # Calculate gaps if we have the data
            woba = row.get('woba')
            slg = row.get('slg')
            xwoba_gap = (xwoba - woba) if (xwoba is not None and woba is not None 
                                           and pd.notna(xwoba) and pd.notna(woba)) else None
            xslg_gap = (xslg - slg) if (xslg is not None and slg is not None 
                                        and pd.notna(xslg) and pd.notna(slg)) else None
            
            # Insert or replace season data
            db.execute("""
                INSERT OR REPLACE INTO statcast_seasons 
                (player_id, year, max_ev, avg_ev, hard_hit_pct, barrel_pct, batted_balls,
                 avg_bat_speed, swing_length, hard_swing_rate, squared_up_rate, whiff_rate, blast_rate,
                 avg_launch_angle, sweet_spot_pct,
                 xwoba, xba, xslg, xwoba_gap, xslg_gap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (player_id, year, max_ev, avg_ev, hard_hit_pct, barrel_pct, batted_balls,
                  avg_bat_speed, swing_length, hard_swing_rate, squared_up_rate, whiff_rate, blast_rate,
                  avg_launch_angle, sweet_spot_pct,
                  xwoba, xba, xslg, xwoba_gap, xslg_gap))
            
            count += 1
            
        except Exception as e:
            print(f"Error loading statcast row: {e}")
            continue
    
    db.commit()
    return count


def load_fangraphs_data(db: MadduxDatabase, df: pd.DataFrame) -> int:
    """
    Load FanGraphs data into database (enhanced with wOBA, BABIP, discipline).
    
    Args:
        db: Database connection
        df: DataFrame with FanGraphs data
        
    Returns:
        Number of rows loaded
    """
    count = 0
    
    for _, row in df.iterrows():
        try:
            # Get FanGraphs ID and player name
            fg_id = int(row.get('playerid', row.get('IDfg', 0)))
            player_name = row.get('Name', 'Unknown')
            
            if fg_id == 0:
                continue
            
            # Get year
            year = int(row.get('year', 0))
            if year == 0:
                continue
            
            # Check if we have a player with this FanGraphs ID
            result = db.query(
                "SELECT id FROM players WHERE fangraphs_id = ?",
                (fg_id,)
            )
            
            if result:
                player_id = result[0][0]
            else:
                # Create new player (will need ID mapping later)
                cursor = db.execute(
                    "INSERT INTO players (fangraphs_id, player_name) VALUES (?, ?)",
                    (fg_id, player_name)
                )
                player_id = cursor.lastrowid
            
            # Get basic stats
            age = row.get('Age')
            pa = row.get('PA')
            avg = row.get('AVG')
            obp = row.get('OBP')
            slg = row.get('SLG')
            ops = row.get('OPS', (obp or 0) + (slg or 0) if obp and slg else None)
            iso = row.get('ISO')
            wrc_plus = row.get('wRC+', row.get('wRC'))
            
            # Get discipline stats
            bb_pct = row.get('BB%', row.get('bb_pct'))
            k_pct = row.get('K%', row.get('k_pct'))
            contact_pct = row.get('Contact%', row.get('contact_pct'))
            chase_rate = row.get('O-Swing%', row.get('chase_rate'))  # Out of zone swing rate
            
            # Get zone swing stats (new)
            z_swing_pct = row.get('Z-Swing%', row.get('z_swing_pct'))  # In-zone swing rate
            z_contact_pct = row.get('Z-Contact%', row.get('z_contact_pct'))  # In-zone contact rate
            
            # Get advanced stats
            woba = row.get('wOBA', row.get('woba'))
            babip = row.get('BABIP', row.get('babip'))
            war = row.get('WAR', row.get('fWAR'))
            
            # Insert or replace season data
            db.execute("""
                INSERT OR REPLACE INTO fangraphs_seasons 
                (player_id, year, age, pa, avg, obp, slg, ops, iso, wrc_plus, 
                 bb_pct, k_pct, contact_pct, chase_rate, z_swing_pct, z_contact_pct,
                 woba, babip, war)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (player_id, year, age, pa, avg, obp, slg, ops, iso, wrc_plus,
                  bb_pct, k_pct, contact_pct, chase_rate, z_swing_pct, z_contact_pct,
                  woba, babip, war))
            
            count += 1
            
        except Exception as e:
            print(f"Error loading fangraphs row: {e}")
            continue
    
    db.commit()
    return count


def create_id_mapping(db: MadduxDatabase) -> int:
    """
    Create player ID mapping between MLB and FanGraphs IDs using pybaseball.
    
    Args:
        db: Database connection
        
    Returns:
        Number of mappings created
    """
    if not PYBASEBALL_AVAILABLE:
        print("pybaseball not available for ID mapping")
        return 0
    
    # Get all MLB IDs from players table
    result = db.query(
        "SELECT mlb_id FROM players WHERE mlb_id IS NOT NULL AND fangraphs_id IS NULL"
    )
    
    if not result:
        return 0
    
    mlb_ids = [row[0] for row in result]
    
    try:
        # Use pybaseball to look up FanGraphs IDs
        mapping = playerid_reverse_lookup(mlb_ids, key_type='mlbam')
        
        count = 0
        for _, row in mapping.iterrows():
            mlb_id = row.get('key_mlbam')
            fg_id = row.get('key_fangraphs')
            
            if pd.notna(mlb_id) and pd.notna(fg_id):
                # Update player record
                db.execute(
                    "UPDATE players SET fangraphs_id = ? WHERE mlb_id = ?",
                    (int(fg_id), int(mlb_id))
                )
                
                # Store in mapping table
                db.execute("""
                    INSERT OR REPLACE INTO player_id_mapping (mlb_id, fangraphs_id, confidence)
                    VALUES (?, ?, 1.0)
                """, (int(mlb_id), int(fg_id)))
                
                count += 1
        
        db.commit()
        return count
        
    except Exception as e:
        print(f"Error creating ID mapping: {e}")
        return 0


def get_player_seasons(db: MadduxDatabase, mlb_id: int = None, 
                       fangraphs_id: int = None) -> pd.DataFrame:
    """
    Get all seasons for a player.
    
    Args:
        db: Database connection
        mlb_id: MLB player ID
        fangraphs_id: FanGraphs player ID
        
    Returns:
        DataFrame with all seasons
    """
    if mlb_id:
        sql = """
            SELECT p.player_name, s.year, s.max_ev, s.hard_hit_pct, s.barrel_pct,
                   f.pa, f.avg, f.obp, f.slg, f.ops, f.wrc_plus
            FROM players p
            LEFT JOIN statcast_seasons s ON p.id = s.player_id
            LEFT JOIN fangraphs_seasons f ON p.id = f.player_id AND s.year = f.year
            WHERE p.mlb_id = ?
            ORDER BY s.year
        """
        return db.query_df(sql, (mlb_id,))
    
    elif fangraphs_id:
        sql = """
            SELECT p.player_name, f.year, s.max_ev, s.hard_hit_pct, s.barrel_pct,
                   f.pa, f.avg, f.obp, f.slg, f.ops, f.wrc_plus
            FROM players p
            LEFT JOIN fangraphs_seasons f ON p.id = f.player_id
            LEFT JOIN statcast_seasons s ON p.id = s.player_id AND f.year = s.year
            WHERE p.fangraphs_id = ?
            ORDER BY f.year
        """
        return db.query_df(sql, (fangraphs_id,))
    
    return pd.DataFrame()


def get_players_by_year(db: MadduxDatabase, year: int) -> pd.DataFrame:
    """
    Get all players with Statcast data for a specific year.
    
    Args:
        db: Database connection
        year: Season year
        
    Returns:
        DataFrame with player data
    """
    sql = """
        SELECT p.id, p.mlb_id, p.fangraphs_id, p.player_name,
               s.max_ev, s.hard_hit_pct, s.barrel_pct, s.avg_ev
        FROM players p
        JOIN statcast_seasons s ON p.id = s.player_id
        WHERE s.year = ?
        ORDER BY s.max_ev DESC
    """
    return db.query_df(sql, (year,))


def filter_by_min_pa(db: MadduxDatabase, year: int, min_pa: int) -> pd.DataFrame:
    """
    Get players meeting minimum PA threshold for a year.
    
    Args:
        db: Database connection
        year: Season year
        min_pa: Minimum plate appearances
        
    Returns:
        DataFrame with qualifying players
    """
    sql = """
        SELECT p.id, p.mlb_id, p.fangraphs_id, p.player_name,
               f.pa, f.avg, f.obp, f.slg, f.ops, f.wrc_plus
        FROM players p
        JOIN fangraphs_seasons f ON p.id = f.player_id
        WHERE f.year = ? AND f.pa >= ?
        ORDER BY f.pa DESC
    """
    return db.query_df(sql, (year, min_pa))


def get_combined_data(db: MadduxDatabase, year: int, min_pa: int = 200) -> pd.DataFrame:
    """
    Get combined Statcast and FanGraphs data for a year (enhanced with all metrics).
    
    Args:
        db: Database connection
        year: Season year
        min_pa: Minimum plate appearances
        
    Returns:
        DataFrame with combined data including bat tracking and expected stats
    """
    sql = """
        SELECT 
            p.id as player_id,
            p.mlb_id,
            p.fangraphs_id,
            p.player_name,
            s.year,
            -- Exit velocity metrics
            s.max_ev,
            s.avg_ev,
            s.hard_hit_pct,
            s.barrel_pct,
            s.batted_balls,
            -- Bat tracking metrics
            s.avg_bat_speed,
            s.swing_length,
            s.hard_swing_rate,
            s.squared_up_rate,
            s.whiff_rate,
            s.blast_rate,
            -- Launch angle metrics
            s.avg_launch_angle,
            s.sweet_spot_pct,
            -- Expected stats
            s.xwoba,
            s.xba,
            s.xslg,
            s.xwoba_gap,
            s.xslg_gap,
            -- FanGraphs basic stats
            f.age,
            f.pa,
            f.avg,
            f.obp,
            f.slg,
            f.ops,
            f.iso,
            f.wrc_plus,
            -- FanGraphs discipline
            f.bb_pct,
            f.k_pct,
            f.contact_pct,
            f.chase_rate,
            -- Zone swing metrics
            f.z_swing_pct,
            f.z_contact_pct,
            -- FanGraphs advanced
            f.woba,
            f.babip,
            f.war
        FROM players p
        JOIN statcast_seasons s ON p.id = s.player_id
        LEFT JOIN fangraphs_seasons f ON p.id = f.player_id AND s.year = f.year
        WHERE s.year = ? AND (f.pa IS NULL OR f.pa >= ?)
        ORDER BY s.max_ev DESC
    """
    return db.query_df(sql, (year, min_pa))


def get_player_features(db: MadduxDatabase, year: int, min_pa: int = 200) -> pd.DataFrame:
    """
    Get player features for ML models.
    
    Args:
        db: Database connection
        year: Season year
        min_pa: Minimum plate appearances
        
    Returns:
        DataFrame with engineered features
    """
    sql = """
        SELECT 
            p.id as player_id,
            p.player_name,
            pf.year,
            pf.underperformance_gap,
            pf.xslg_gap_scaled,
            pf.babip_gap,
            pf.iso_gap,
            pf.combined_luck_index,
            pf.deviation_from_baseline,
            pf.career_peak_deviation,
            pf.improvement_momentum,
            pf.ops_trend_2yr,
            pf.years_from_peak,
            pf.bat_speed_zscore,
            pf.swing_length_factor,
            pf.squared_up_quality,
            pf.whiff_adjusted_power,
            pf.quality_zscore,
            pf.hard_hit_zscore,
            pf.age_factor,
            pf.pa_weight,
            pf.original_maddux_score,
            pf.delta_launch_angle,
            pf.delta_sweet_spot,
            pf.launch_angle_zscore,
            pf.delta_z_swing,
            pf.discipline_improvement,
            f.pa,
            f.ops,
            f.wrc_plus,
            f.age
        FROM player_features pf
        JOIN players p ON pf.player_id = p.id
        LEFT JOIN fangraphs_seasons f ON pf.player_id = f.player_id AND pf.year = f.year
        WHERE pf.year = ? AND (f.pa IS NULL OR f.pa >= ?)
        ORDER BY pf.deviation_from_baseline DESC
    """
    return db.query_df(sql, (year, min_pa))


def store_player_features(db: MadduxDatabase, features_df: pd.DataFrame) -> int:
    """
    Store calculated player features in database.
    
    Args:
        db: Database connection
        features_df: DataFrame with player features
        
    Returns:
        Number of rows stored
    """
    count = 0
    for _, row in features_df.iterrows():
        try:
            db.execute("""
                INSERT OR REPLACE INTO player_features 
                (player_id, year, underperformance_gap, xslg_gap_scaled, babip_gap, iso_gap,
                 combined_luck_index, deviation_from_baseline, career_peak_deviation,
                 improvement_momentum, ops_trend_2yr, years_from_peak, bat_speed_zscore,
                 swing_length_factor, squared_up_quality, whiff_adjusted_power,
                 quality_zscore, hard_hit_zscore, age_factor, pa_weight, original_maddux_score,
                 delta_launch_angle, delta_sweet_spot, launch_angle_zscore,
                 delta_z_swing, discipline_improvement)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get('player_id'),
                row.get('year'),
                row.get('underperformance_gap'),
                row.get('xslg_gap_scaled'),
                row.get('babip_gap'),
                row.get('iso_gap'),
                row.get('combined_luck_index'),
                row.get('deviation_from_baseline'),
                row.get('career_peak_deviation'),
                row.get('improvement_momentum'),
                row.get('ops_trend_2yr'),
                row.get('years_from_peak'),
                row.get('bat_speed_zscore'),
                row.get('swing_length_factor'),
                row.get('squared_up_quality'),
                row.get('whiff_adjusted_power'),
                row.get('quality_zscore'),
                row.get('hard_hit_zscore'),
                row.get('age_factor'),
                row.get('pa_weight'),
                row.get('original_maddux_score'),
                row.get('delta_launch_angle'),
                row.get('delta_sweet_spot'),
                row.get('launch_angle_zscore'),
                row.get('delta_z_swing'),
                row.get('discipline_improvement')
            ))
            count += 1
        except Exception as e:
            print(f"Error storing features: {e}")
            continue
    
    db.commit()
    return count


def store_model_prediction(db: MadduxDatabase, prediction: Dict[str, Any]) -> bool:
    """
    Store a model prediction with confidence intervals.
    
    Args:
        db: Database connection
        prediction: Dict with prediction data
        
    Returns:
        True if successful
    """
    try:
        db.execute("""
            INSERT OR REPLACE INTO model_predictions
            (player_id, prediction_year, target_year, model_type, target_variable,
             predicted_value, confidence_low, confidence_high, actual_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction['player_id'],
            prediction['prediction_year'],
            prediction['target_year'],
            prediction['model_type'],
            prediction['target_variable'],
            prediction['predicted_value'],
            prediction.get('confidence_low'),
            prediction.get('confidence_high'),
            prediction.get('actual_value')
        ))
        db.commit()
        return True
    except Exception as e:
        print(f"Error storing prediction: {e}")
        return False


def load_bat_tracking_data(db: MadduxDatabase, df: pd.DataFrame, year: int) -> int:
    """
    Load bat tracking data and merge with existing Statcast records.
    
    Args:
        db: Database connection
        df: DataFrame with bat tracking data
        year: Year for the data
        
    Returns:
        Number of rows updated
    """
    count = 0
    for _, row in df.iterrows():
        try:
            mlb_id = int(row.get('player_id', 0))
            if mlb_id == 0:
                continue
            
            # Get internal player_id
            result = db.query("SELECT id FROM players WHERE mlb_id = ?", (mlb_id,))
            if not result:
                continue
            
            player_id = result[0][0]
            
            # Update bat tracking columns
            db.execute("""
                UPDATE statcast_seasons 
                SET avg_bat_speed = ?,
                    swing_length = ?,
                    hard_swing_rate = ?,
                    squared_up_rate = ?,
                    whiff_rate = ?,
                    blast_rate = ?
                WHERE player_id = ? AND year = ?
            """, (
                row.get('avg_bat_speed'),
                row.get('swing_length'),
                row.get('hard_swing_rate'),
                row.get('squared_up_per_swing', row.get('squared_up_rate')),
                row.get('whiff_per_swing', row.get('whiff_rate')),
                row.get('blast_per_swing', row.get('blast_rate')),
                player_id,
                year
            ))
            count += 1
            
        except Exception as e:
            print(f"Error loading bat tracking: {e}")
            continue
    
    db.commit()
    return count


def load_all_data(db: MadduxDatabase, data_dir: Path = None) -> Dict[str, int]:
    """
    Load all data from CSV files into database (enhanced with bat tracking).
    
    Args:
        db: Database connection
        data_dir: Directory containing data files
        
    Returns:
        Dictionary with counts of loaded data
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    
    counts = {'statcast': 0, 'fangraphs': 0, 'bat_tracking': 0}
    
    # Load combined Statcast data
    statcast_file = data_dir / "all_statcast.csv"
    if statcast_file.exists():
        print(f"Loading Statcast data from {statcast_file}...")
        df = pd.read_csv(statcast_file)
        counts['statcast'] = load_statcast_data(db, df)
        print(f"  Loaded {counts['statcast']} Statcast records")
    
    # Load combined FanGraphs data
    fangraphs_file = data_dir / "all_fangraphs.csv"
    if fangraphs_file.exists():
        print(f"Loading FanGraphs data from {fangraphs_file}...")
        df = pd.read_csv(fangraphs_file)
        counts['fangraphs'] = load_fangraphs_data(db, df)
        print(f"  Loaded {counts['fangraphs']} FanGraphs records")
    
    # Load bat tracking data by year (new)
    bat_tracking_total = 0
    for year_dir in sorted(data_dir.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            year = int(year_dir.name)
            bat_file = year_dir / "bat_tracking.csv"
            if bat_file.exists():
                print(f"Loading bat tracking data for {year}...")
                df = pd.read_csv(bat_file)
                updated = load_bat_tracking_data(db, df, year)
                bat_tracking_total += updated
                print(f"  Updated {updated} records")
    
    # Also check for test_bat_tracking.csv in parent directories
    parent_bat_tracking = data_dir.parent.parent / "maddux_test_1" / "data" / "test_bat_tracking.csv"
    if parent_bat_tracking.exists():
        print(f"Loading bat tracking from test_1 data...")
        df = pd.read_csv(parent_bat_tracking)
        # Assume it's 2024 data based on maddux_test_1
        updated = load_bat_tracking_data(db, df, 2024)
        bat_tracking_total += updated
        print(f"  Updated {updated} records")
    
    counts['bat_tracking'] = bat_tracking_total
    
    # Create ID mapping
    print("Creating player ID mapping...")
    mapped = create_id_mapping(db)
    print(f"  Mapped {mapped} players")
    
    return counts


def migrate_schema(db: MadduxDatabase) -> bool:
    """
    Migrate existing database to new schema by adding missing columns.
    Safe to run multiple times.
    
    Args:
        db: Database connection
        
    Returns:
        True if migration successful
    """
    try:
        # Get existing columns in statcast_seasons
        existing_cols = set(db.get_table_columns('statcast_seasons'))
        
        # New columns to add to statcast_seasons
        new_statcast_cols = [
            ('avg_bat_speed', 'REAL'),
            ('swing_length', 'REAL'),
            ('hard_swing_rate', 'REAL'),
            ('squared_up_rate', 'REAL'),
            ('whiff_rate', 'REAL'),
            ('blast_rate', 'REAL'),
            ('avg_launch_angle', 'REAL'),
            ('sweet_spot_pct', 'REAL'),
            ('xwoba', 'REAL'),
            ('xba', 'REAL'),
            ('xslg', 'REAL'),
            ('xwoba_gap', 'REAL'),
            ('xslg_gap', 'REAL'),
        ]
        
        for col_name, col_type in new_statcast_cols:
            if col_name not in existing_cols:
                print(f"  Adding column statcast_seasons.{col_name}")
                db.execute(f"ALTER TABLE statcast_seasons ADD COLUMN {col_name} {col_type}")
        
        # Get existing columns in fangraphs_seasons
        existing_fg_cols = set(db.get_table_columns('fangraphs_seasons'))
        
        # New columns to add to fangraphs_seasons
        new_fg_cols = [
            ('age', 'INTEGER'),
            ('contact_pct', 'REAL'),
            ('chase_rate', 'REAL'),
            ('z_swing_pct', 'REAL'),
            ('z_contact_pct', 'REAL'),
            ('woba', 'REAL'),
            ('babip', 'REAL'),
            ('war', 'REAL'),
        ]
        
        for col_name, col_type in new_fg_cols:
            if col_name not in existing_fg_cols:
                print(f"  Adding column fangraphs_seasons.{col_name}")
                db.execute(f"ALTER TABLE fangraphs_seasons ADD COLUMN {col_name} {col_type}")
        
        # Create new tables if they don't exist
        tables = set(db.get_tables())
        
        if 'player_features' not in tables:
            print("  Creating player_features table")
            db.execute("""
                CREATE TABLE player_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,
                    year INTEGER NOT NULL,
                    underperformance_gap REAL,
                    xslg_gap_scaled REAL,
                    babip_gap REAL,
                    iso_gap REAL,
                    combined_luck_index REAL,
                    deviation_from_baseline REAL,
                    career_peak_deviation REAL,
                    improvement_momentum REAL,
                    ops_trend_2yr REAL,
                    years_from_peak INTEGER,
                    bat_speed_zscore REAL,
                    swing_length_factor REAL,
                    squared_up_quality REAL,
                    whiff_adjusted_power REAL,
                    quality_zscore REAL,
                    hard_hit_zscore REAL,
                    age_factor REAL,
                    pa_weight REAL,
                    original_maddux_score REAL,
                    -- Launch angle features (new)
                    delta_launch_angle REAL,
                    delta_sweet_spot REAL,
                    launch_angle_zscore REAL,
                    -- Zone swing features (new)
                    delta_z_swing REAL,
                    discipline_improvement REAL,
                    FOREIGN KEY (player_id) REFERENCES players(id),
                    UNIQUE(player_id, year)
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS idx_features_year ON player_features(year)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_features_player ON player_features(player_id)")
        else:
            # Add new columns to existing player_features table
            existing_feature_cols = set(db.get_table_columns('player_features'))
            new_feature_cols = [
                ('delta_launch_angle', 'REAL'),
                ('delta_sweet_spot', 'REAL'),
                ('launch_angle_zscore', 'REAL'),
                ('delta_z_swing', 'REAL'),
                ('discipline_improvement', 'REAL'),
            ]
            for col_name, col_type in new_feature_cols:
                if col_name not in existing_feature_cols:
                    print(f"  Adding column player_features.{col_name}")
                    db.execute(f"ALTER TABLE player_features ADD COLUMN {col_name} {col_type}")
        
        if 'model_predictions' not in tables:
            print("  Creating model_predictions table")
            db.execute("""
                CREATE TABLE model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,
                    prediction_year INTEGER NOT NULL,
                    target_year INTEGER NOT NULL,
                    model_type TEXT NOT NULL,
                    target_variable TEXT NOT NULL,
                    predicted_value REAL,
                    confidence_low REAL,
                    confidence_high REAL,
                    actual_value REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players(id),
                    UNIQUE(player_id, prediction_year, target_year, model_type, target_variable)
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_year ON model_predictions(target_year)")
        
        db.commit()
        print("Schema migration complete")
        return True
        
    except Exception as e:
        print(f"Migration error: {e}")
        return False


def main():
    """Initialize database with all data (enhanced v2)."""
    print("=" * 60)
    print("MADDUX Phase 1 - Enhanced Database Setup (v2)")
    print("=" * 60)
    
    # Create database
    db = MadduxDatabase()
    
    # Create schema (or migrate existing)
    tables = db.get_tables()
    if 'players' in tables:
        print("\nExisting database detected - running migration...")
        migrate_schema(db)
    else:
        print("\nCreating new database schema...")
        create_schema(db)
    
    # Load all data
    print("\n" + "-" * 60)
    counts = load_all_data(db)
    
    # Summary
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)
    
    # Count records
    players = db.query("SELECT COUNT(*) FROM players")[0][0]
    statcast = db.query("SELECT COUNT(*) FROM statcast_seasons")[0][0]
    fangraphs = db.query("SELECT COUNT(*) FROM fangraphs_seasons")[0][0]
    
    print(f"Players: {players}")
    print(f"Statcast seasons: {statcast}")
    print(f"FanGraphs seasons: {fangraphs}")
    
    # Count bat tracking data
    bat_tracking_count = db.query("""
        SELECT COUNT(*) FROM statcast_seasons 
        WHERE avg_bat_speed IS NOT NULL
    """)[0][0]
    print(f"Records with bat tracking: {bat_tracking_count}")
    
    # Count expected stats
    xstats_count = db.query("""
        SELECT COUNT(*) FROM statcast_seasons 
        WHERE xwoba IS NOT NULL
    """)[0][0]
    print(f"Records with expected stats: {xstats_count}")
    
    # Check player_features
    try:
        features_count = db.query("SELECT COUNT(*) FROM player_features")[0][0]
        print(f"Player features: {features_count}")
    except Exception:
        print("Player features: 0 (run feature_engineering.py)")
    
    # Count by year
    print("\nRecords by year:")
    result = db.query("""
        SELECT year, COUNT(*) as count 
        FROM statcast_seasons 
        GROUP BY year 
        ORDER BY year
    """)
    for year, count in result:
        # Check bat tracking coverage for this year
        bt_count = db.query("""
            SELECT COUNT(*) FROM statcast_seasons 
            WHERE year = ? AND avg_bat_speed IS NOT NULL
        """, (year,))[0][0]
        bt_pct = (bt_count / count * 100) if count > 0 else 0
        print(f"  {year}: {count} players ({bt_pct:.0f}% with bat tracking)")
    
    db.close()
    print("\n" + "=" * 60)
    print("Database setup complete (enhanced v2)!")
    print("=" * 60)


if __name__ == "__main__":
    main()

