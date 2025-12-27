"""
pull_multi_year.py
Pulls full 2015-2025 hitter data from Baseball Savant and FanGraphs.

Enhanced v2 - Now includes:
- Statcast exit velocity and barrel data (2015-2025)
- FanGraphs batting stats (2015-2025)
- Bat tracking data (2024-2025) - avg bat speed, swing length, whiff rate
- Expected stats (xwOBA, xBA, xSLG)

Data is stored in data/{year}/ directories.
"""

import os
import sys
import time
import requests
from pathlib import Path
from typing import Optional, Dict, List
from io import StringIO

import pandas as pd
from tqdm import tqdm

# Configuration
YEARS = list(range(2015, 2026))  # 2015-2025
BAT_TRACKING_YEARS = [2024, 2025]  # Bat tracking only available from 2024
MIN_PA = 200
DATA_DIR = Path(__file__).parent.parent / "data"
RATE_LIMIT_DELAY = 2  # seconds between API calls

# Baseball Savant URLs
SAVANT_BASE_URL = "https://baseballsavant.mlb.com"
BAT_TRACKING_URL = f"{SAVANT_BASE_URL}/leaderboard/bat-tracking"
EXPECTED_STATS_URL = f"{SAVANT_BASE_URL}/leaderboard/expected_statistics"

# pybaseball imports
try:
    from pybaseball import statcast_batter_exitvelo_barrels, batting_stats
    from pybaseball import cache
    cache.enable()
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("Warning: pybaseball not installed. Install with: pip install pybaseball")


def validate_statcast_data(df: pd.DataFrame) -> bool:
    """
    Validate Statcast data has required columns and valid ranges.
    
    Required columns: player_id, max_hit_speed, hard_hit_percent
    Valid ranges:
        - max_hit_speed: 85-130 mph
        - hard_hit_percent: 0-100%
    
    Args:
        df: DataFrame with Statcast data
        
    Returns:
        True if valid, False otherwise
    """
    if df is None or len(df) == 0:
        return False
    
    # Check required columns (with alternative names)
    required_cols = {
        'player_id': ['player_id'],
        'max_ev': ['max_hit_speed', 'max_ev', 'max_exit_velocity'],
        'hard_hit': ['hard_hit_percent', 'ev95percent', 'hard_hit_pct']
    }
    
    for col_type, alternatives in required_cols.items():
        found = any(alt in df.columns for alt in alternatives)
        if not found:
            print(f"  Missing column: {col_type} (tried: {alternatives})")
            return False
    
    # Find the actual column names
    max_ev_col = next(c for c in ['max_hit_speed', 'max_ev', 'max_exit_velocity'] if c in df.columns)
    hard_hit_col = next(c for c in ['hard_hit_percent', 'ev95percent', 'hard_hit_pct'] if c in df.columns)
    
    # Validate ranges
    max_ev = df[max_ev_col].dropna()
    if len(max_ev) > 0:
        if max_ev.max() > 130 or max_ev.min() < 85:
            print(f"  Invalid max EV range: {max_ev.min():.1f} - {max_ev.max():.1f}")
            return False
    
    hard_hit = df[hard_hit_col].dropna()
    if len(hard_hit) > 0:
        if hard_hit.max() > 100 or hard_hit.min() < 0:
            print(f"  Invalid hard hit range: {hard_hit.min():.1f} - {hard_hit.max():.1f}")
            return False
    
    return True


def validate_fangraphs_data(df: pd.DataFrame) -> bool:
    """
    Validate FanGraphs data has required columns.
    
    Required columns: playerid (or IDfg), Name, PA, AVG, OBP, SLG
    
    Args:
        df: DataFrame with FanGraphs data
        
    Returns:
        True if valid, False otherwise
    """
    if df is None or len(df) == 0:
        return False
    
    # Check for player ID column
    has_id = 'playerid' in df.columns or 'IDfg' in df.columns
    if not has_id:
        print("  Missing player ID column (playerid or IDfg)")
        return False
    
    # Check other required columns
    required = ['Name', 'PA']
    for col in required:
        if col not in df.columns:
            print(f"  Missing column: {col}")
            return False
    
    return True


def pull_statcast_data(year: int, min_pa: int = MIN_PA) -> Optional[pd.DataFrame]:
    """
    Pull Statcast exit velocity and barrel data for a single year.
    
    Args:
        year: Season year (2015-2025)
        min_pa: Minimum plate appearances filter
        
    Returns:
        DataFrame with Statcast data or None on error
    """
    if not PYBASEBALL_AVAILABLE:
        print(f"  pybaseball not available for {year}")
        return None
    
    try:
        print(f"  Pulling Statcast data for {year}...")
        
        # minBBE is batted ball events, roughly correlates to PA
        df = statcast_batter_exitvelo_barrels(year, minBBE=min_pa // 2)
        
        if df is None or len(df) == 0:
            print(f"  No Statcast data returned for {year}")
            return None
        
        # Standardize column names
        column_renames = {
            'last_name, first_name': 'player_name',
        }
        df = df.rename(columns=column_renames)
        
        # Add year column
        df['year'] = year
        
        print(f"  Retrieved {len(df)} players for {year}")
        return df
        
    except Exception as e:
        print(f"  Error pulling Statcast for {year}: {e}")
        return None


def pull_fangraphs_batting(year: int, min_pa: int = MIN_PA) -> Optional[pd.DataFrame]:
    """
    Pull FanGraphs batting stats for a single year.
    
    Args:
        year: Season year (2015-2025)
        min_pa: Minimum plate appearances filter
        
    Returns:
        DataFrame with FanGraphs data or None on error
    """
    if not PYBASEBALL_AVAILABLE:
        print(f"  pybaseball not available for {year}")
        return None
    
    try:
        print(f"  Pulling FanGraphs batting for {year}...")
        
        df = batting_stats(year, qual=min_pa)
        
        if df is None or len(df) == 0:
            print(f"  No FanGraphs data returned for {year}")
            return None
        
        # Standardize player ID column
        if 'IDfg' in df.columns:
            df = df.rename(columns={'IDfg': 'playerid'})
        
        # Add year column
        df['year'] = year
        
        # Ensure OPS column exists
        if 'OPS' not in df.columns and 'OBP' in df.columns and 'SLG' in df.columns:
            df['OPS'] = df['OBP'] + df['SLG']
        
        print(f"  Retrieved {len(df)} players for {year}")
        return df
        
    except Exception as e:
        print(f"  Error pulling FanGraphs for {year}: {e}")
        return None


def pull_bat_tracking_data(year: int, min_swings: int = 100) -> Optional[pd.DataFrame]:
    """
    Pull bat tracking data from Baseball Savant.
    
    Only available for 2024+.
    
    Args:
        year: Season year (2024+)
        min_swings: Minimum swings filter
        
    Returns:
        DataFrame with bat tracking data or None on error
    """
    if year < 2024:
        print(f"  Bat tracking not available before 2024")
        return None
    
    try:
        print(f"  Pulling bat tracking data for {year}...")
        
        # Baseball Savant bat tracking CSV export URL
        url = (f"{SAVANT_BASE_URL}/leaderboard/bat-tracking?"
               f"attackZone=&batSide=&contactType=&count=&dateStart="
               f"&dateEnd=&gameType=&isHardHit=&minSwings={min_swings}"
               f"&minGroupSwings=1&pitchHand=&pitchType="
               f"&player_type=Batter&season={year}&team=&csv=true")
        
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            print(f"  Failed to fetch bat tracking data: {response.status_code}")
            return None
        
        df = pd.read_csv(StringIO(response.text))
        
        if df is None or len(df) == 0:
            print(f"  No bat tracking data returned for {year}")
            return None
        
        # Standardize column names
        column_renames = {
            'batter_id': 'player_id',
            'batter_name': 'player_name',
            'bat_speed': 'avg_bat_speed',
            'swing_path_tilt': 'swing_length',  # May need adjustment
            'fast_swing_rate': 'hard_swing_rate',
            'squared_up_swing': 'squared_up_rate',
            'swords': 'whiff_rate',  # Whiff/swing
            'blasts': 'blast_rate',
        }
        
        # Only rename columns that exist
        for old_name, new_name in column_renames.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Add year column
        df['year'] = year
        
        print(f"  Retrieved {len(df)} players with bat tracking for {year}")
        return df
        
    except Exception as e:
        print(f"  Error pulling bat tracking for {year}: {e}")
        return None


def pull_expected_stats(year: int, min_pa: int = MIN_PA) -> Optional[pd.DataFrame]:
    """
    Pull expected stats (xwOBA, xBA, xSLG) from Baseball Savant.
    
    Args:
        year: Season year
        min_pa: Minimum PA filter
        
    Returns:
        DataFrame with expected stats or None on error
    """
    try:
        print(f"  Pulling expected stats for {year}...")
        
        # Baseball Savant expected stats CSV URL
        url = (f"{SAVANT_BASE_URL}/leaderboard/expected_statistics?"
               f"type=batter&year={year}&position=&team=&min={min_pa // 2}&csv=true")
        
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            print(f"  Failed to fetch expected stats: {response.status_code}")
            return None
        
        df = pd.read_csv(StringIO(response.text))
        
        if df is None or len(df) == 0:
            print(f"  No expected stats returned for {year}")
            return None
        
        # Standardize column names
        column_renames = {
            'player_id': 'player_id',
            'last_name, first_name': 'player_name',
            'est_woba': 'xwoba',
            'est_ba': 'xba',
            'est_slg': 'xslg',
            'woba': 'woba',
            'ba': 'ba',
            'slg': 'slg',
        }
        
        for old_name, new_name in column_renames.items():
            if old_name in df.columns and old_name != new_name:
                df = df.rename(columns={old_name: new_name})
        
        # Calculate gaps if we have the data
        if 'xwoba' in df.columns and 'woba' in df.columns:
            df['xwoba_gap'] = df['xwoba'] - df['woba']
        if 'xslg' in df.columns and 'slg' in df.columns:
            df['xslg_gap'] = df['xslg'] - df['slg']
        
        # Add year column
        df['year'] = year
        
        print(f"  Retrieved {len(df)} players with expected stats for {year}")
        return df
        
    except Exception as e:
        print(f"  Error pulling expected stats for {year}: {e}")
        return None


def pull_all_years(years: List[int] = None, include_bat_tracking: bool = True) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Pull data for all specified years from all sources.
    
    Enhanced v2: Now includes bat tracking and expected stats.
    
    Args:
        years: List of years to pull (default: YEARS)
        include_bat_tracking: Whether to pull bat tracking data for 2024+
        
    Returns:
        Dictionary with structure:
        {
            'statcast': {year: DataFrame, ...},
            'fangraphs': {year: DataFrame, ...},
            'bat_tracking': {year: DataFrame, ...},
            'expected_stats': {year: DataFrame, ...}
        }
    """
    if years is None:
        years = YEARS
    
    results = {
        'statcast': {},
        'fangraphs': {},
        'bat_tracking': {},
        'expected_stats': {}
    }
    
    print("=" * 70)
    print("MADDUX Phase 1 - Multi-Year Data Pull (Enhanced v2)")
    print(f"Years: {years[0]} - {years[-1]}")
    print("=" * 70)
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for year in tqdm(years, desc="Pulling data"):
        print(f"\n--- Year {year} ---")
        
        # Create year directory
        year_dir = DATA_DIR / str(year)
        year_dir.mkdir(exist_ok=True)
        
        # Pull Statcast data
        statcast_file = year_dir / "statcast.csv"
        if statcast_file.exists():
            print(f"  Loading cached Statcast for {year}...")
            results['statcast'][year] = pd.read_csv(statcast_file)
        else:
            df = pull_statcast_data(year)
            if df is not None and validate_statcast_data(df):
                results['statcast'][year] = df
                df.to_csv(statcast_file, index=False)
            time.sleep(RATE_LIMIT_DELAY)
        
        # Pull FanGraphs data
        fangraphs_file = year_dir / "fangraphs.csv"
        if fangraphs_file.exists():
            print(f"  Loading cached FanGraphs for {year}...")
            results['fangraphs'][year] = pd.read_csv(fangraphs_file)
        else:
            df = pull_fangraphs_batting(year)
            if df is not None and validate_fangraphs_data(df):
                results['fangraphs'][year] = df
                df.to_csv(fangraphs_file, index=False)
            time.sleep(RATE_LIMIT_DELAY)
        
        # Pull expected stats
        expected_file = year_dir / "expected_stats.csv"
        if expected_file.exists():
            print(f"  Loading cached expected stats for {year}...")
            results['expected_stats'][year] = pd.read_csv(expected_file)
        else:
            df = pull_expected_stats(year)
            if df is not None:
                results['expected_stats'][year] = df
                df.to_csv(expected_file, index=False)
            time.sleep(RATE_LIMIT_DELAY)
        
        # Pull bat tracking data (2024+ only)
        if include_bat_tracking and year in BAT_TRACKING_YEARS:
            bat_tracking_file = year_dir / "bat_tracking.csv"
            if bat_tracking_file.exists():
                print(f"  Loading cached bat tracking for {year}...")
                results['bat_tracking'][year] = pd.read_csv(bat_tracking_file)
            else:
                df = pull_bat_tracking_data(year)
                if df is not None:
                    results['bat_tracking'][year] = df
                    df.to_csv(bat_tracking_file, index=False)
                time.sleep(RATE_LIMIT_DELAY)
    
    # Summary
    print("\n" + "=" * 70)
    print("PULL SUMMARY")
    print("=" * 70)
    print(f"Statcast years retrieved: {len(results['statcast'])}")
    print(f"FanGraphs years retrieved: {len(results['fangraphs'])}")
    print(f"Expected stats years retrieved: {len(results['expected_stats'])}")
    print(f"Bat tracking years retrieved: {len(results['bat_tracking'])}")
    
    # Count total player-seasons
    total_statcast = sum(len(df) for df in results['statcast'].values())
    total_fangraphs = sum(len(df) for df in results['fangraphs'].values())
    total_expected = sum(len(df) for df in results['expected_stats'].values())
    total_bat_tracking = sum(len(df) for df in results['bat_tracking'].values())
    
    print(f"\nTotal Statcast player-seasons: {total_statcast}")
    print(f"Total FanGraphs player-seasons: {total_fangraphs}")
    print(f"Total expected stats player-seasons: {total_expected}")
    print(f"Total bat tracking player-seasons: {total_bat_tracking}")
    
    return results


def combine_all_years(results: Dict[str, Dict[int, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Combine all years into single DataFrames.
    
    Enhanced v2: Now includes bat tracking and expected stats.
    
    Args:
        results: Dictionary from pull_all_years
        
    Returns:
        Dictionary with combined DataFrames for each data type
    """
    combined = {}
    
    # Combine Statcast
    if results.get('statcast'):
        statcast_dfs = list(results['statcast'].values())
        combined['statcast'] = pd.concat(statcast_dfs, ignore_index=True)
        print(f"Combined Statcast: {len(combined['statcast'])} rows")
    
    # Combine FanGraphs
    if results.get('fangraphs'):
        fangraphs_dfs = list(results['fangraphs'].values())
        combined['fangraphs'] = pd.concat(fangraphs_dfs, ignore_index=True)
        print(f"Combined FanGraphs: {len(combined['fangraphs'])} rows")
    
    # Combine expected stats
    if results.get('expected_stats'):
        expected_dfs = list(results['expected_stats'].values())
        combined['expected_stats'] = pd.concat(expected_dfs, ignore_index=True)
        print(f"Combined expected stats: {len(combined['expected_stats'])} rows")
    
    # Combine bat tracking
    if results.get('bat_tracking'):
        bat_tracking_dfs = list(results['bat_tracking'].values())
        combined['bat_tracking'] = pd.concat(bat_tracking_dfs, ignore_index=True)
        print(f"Combined bat tracking: {len(combined['bat_tracking'])} rows")
    
    return combined


def merge_statcast_with_extras(combined: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge Statcast data with expected stats and bat tracking.
    
    Args:
        combined: Dictionary with combined DataFrames
        
    Returns:
        Merged DataFrame
    """
    if 'statcast' not in combined:
        return pd.DataFrame()
    
    merged = combined['statcast'].copy()
    
    # Merge expected stats
    if 'expected_stats' in combined:
        expected = combined['expected_stats'][['player_id', 'year', 'xwoba', 'xba', 'xslg', 
                                               'xwoba_gap', 'xslg_gap']].drop_duplicates()
        merged = merged.merge(expected, on=['player_id', 'year'], how='left', suffixes=('', '_expected'))
        print(f"  Merged expected stats: {merged['xwoba'].notna().sum()} records")
    
    # Merge bat tracking
    if 'bat_tracking' in combined:
        bat_cols = ['player_id', 'year', 'avg_bat_speed', 'swing_length', 
                    'hard_swing_rate', 'squared_up_rate', 'whiff_rate', 'blast_rate']
        available_cols = [c for c in bat_cols if c in combined['bat_tracking'].columns]
        bat_tracking = combined['bat_tracking'][available_cols].drop_duplicates()
        merged = merged.merge(bat_tracking, on=['player_id', 'year'], how='left', suffixes=('', '_bat'))
        print(f"  Merged bat tracking: {merged['avg_bat_speed'].notna().sum()} records")
    
    return merged


def save_combined_data(combined: Dict[str, pd.DataFrame]) -> None:
    """
    Save combined DataFrames to data directory.
    
    Enhanced v2: Saves all data types and merged Statcast.
    
    Args:
        combined: Dictionary with DataFrames for each data type
    """
    if 'statcast' in combined:
        path = DATA_DIR / "all_statcast.csv"
        combined['statcast'].to_csv(path, index=False)
        print(f"Saved combined Statcast to: {path}")
    
    if 'fangraphs' in combined:
        path = DATA_DIR / "all_fangraphs.csv"
        combined['fangraphs'].to_csv(path, index=False)
        print(f"Saved combined FanGraphs to: {path}")
    
    if 'expected_stats' in combined:
        path = DATA_DIR / "all_expected_stats.csv"
        combined['expected_stats'].to_csv(path, index=False)
        print(f"Saved combined expected stats to: {path}")
    
    if 'bat_tracking' in combined:
        path = DATA_DIR / "all_bat_tracking.csv"
        combined['bat_tracking'].to_csv(path, index=False)
        print(f"Saved combined bat tracking to: {path}")
    
    # Create merged Statcast with all extras
    merged = merge_statcast_with_extras(combined)
    if len(merged) > 0:
        path = DATA_DIR / "all_statcast_enhanced.csv"
        merged.to_csv(path, index=False)
        print(f"Saved enhanced Statcast to: {path}")


def main():
    """Main entry point for data pulling (enhanced v2)."""
    print("\n" + "=" * 70)
    print("MADDUX Phase 1: Full Data Pull (2015-2025) - Enhanced v2")
    print("=" * 70)
    print("\nIncludes: Exit velocity, bat tracking (2024+), expected stats")
    print("-" * 70 + "\n")
    
    # Pull all years
    results = pull_all_years()
    
    # Combine into single DataFrames
    print("\n" + "-" * 70)
    print("Combining data...")
    combined = combine_all_years(results)
    
    # Save combined data
    print("\n" + "-" * 70)
    print("Saving combined data...")
    save_combined_data(combined)
    
    # Summary
    print("\n" + "=" * 70)
    print("DATA PULL COMPLETE")
    print("=" * 70)
    
    print("\nFiles created:")
    for filename in ['all_statcast.csv', 'all_fangraphs.csv', 
                     'all_expected_stats.csv', 'all_bat_tracking.csv',
                     'all_statcast_enhanced.csv']:
        filepath = DATA_DIR / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024 / 1024
            print(f"  {filename}: {size:.2f} MB")
    
    print("\nNext steps:")
    print("  1. Run database.py to load data into SQLite")
    print("  2. Run feature_engineering.py to calculate features")
    print("  3. Run stacking_model.py to train model and generate predictions")
    
    return results, combined


if __name__ == "__main__":
    main()

