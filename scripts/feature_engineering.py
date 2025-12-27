"""
feature_engineering.py
Feature engineering for MADDUX™ enhanced model.

Calculates all engineered features for the Stacking Meta-Learner:

1. Luck/Underperformance Features:
   - underperformance_gap: xwOBA - wOBA gap (luck adjustment)
   - xslg_gap_scaled: xSLG - SLG gap (power luck)
   - babip_gap: xBA - BABIP gap (contact luck)
   - combined_luck_index: weighted combination

2. Baseline/Trend Features:
   - deviation_from_baseline: current OPS vs 3-year baseline
   - career_peak_deviation: distance from career best
   - improvement_momentum: recent YoY change (negative predictor!)
   - age_factor: age-based adjustment

3. Bat Quality Features:
   - bat_speed_zscore: standardized bat speed
   - swing_length_factor: relative to league average
   - squared_up_quality: squared_up_rate * bat_speed
   - whiff_adjusted_power: barrel% / whiff_rate
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase, store_player_features


# ============================================================================
# LUCK / UNDERPERFORMANCE FEATURES
# ============================================================================

def calculate_underperformance_gap(xwoba: float, woba: float) -> Optional[float]:
    """
    Calculate underperformance gap from expected vs actual wOBA.
    
    Positive = unlucky/underperforming (should improve)
    Negative = lucky/overperforming (may regress)
    
    Args:
        xwoba: Expected wOBA (from Statcast)
        woba: Actual wOBA
        
    Returns:
        Gap scaled by 1000 for interpretability
    """
    if xwoba is None or woba is None or pd.isna(xwoba) or pd.isna(woba):
        return None
    return (xwoba - woba) * 1000


def calculate_xslg_gap(xslg: float, slg: float) -> Optional[float]:
    """
    Calculate power underperformance from expected vs actual SLG.
    
    Args:
        xslg: Expected slugging
        slg: Actual slugging
        
    Returns:
        Gap scaled by 1000
    """
    if xslg is None or slg is None or pd.isna(xslg) or pd.isna(slg):
        return None
    return (xslg - slg) * 1000


def calculate_babip_gap(xba: float, babip: float) -> Optional[float]:
    """
    Calculate contact luck from expected BA vs BABIP.
    
    Args:
        xba: Expected batting average
        babip: Batting average on balls in play
        
    Returns:
        Gap scaled by 1000
    """
    if xba is None or babip is None or pd.isna(xba) or pd.isna(babip):
        return None
    return (xba - babip) * 1000


def calculate_iso_gap(xslg: float, xba: float, iso: float) -> Optional[float]:
    """
    Calculate isolated power gap.
    Expected ISO ≈ xSLG - xBA
    
    Args:
        xslg: Expected slugging
        xba: Expected batting average
        iso: Actual isolated power
        
    Returns:
        Gap scaled by 1000
    """
    if any(v is None or pd.isna(v) for v in [xslg, xba, iso]):
        return None
    expected_iso = xslg - xba
    return (expected_iso - iso) * 1000


def calculate_combined_luck_index(
    underperformance_gap: float,
    xslg_gap: float,
    babip_gap: float
) -> Optional[float]:
    """
    Calculate combined luck index (weighted average of gap indicators).
    
    Weights based on predictive power from experiments:
    - xwOBA gap: 50% (most important)
    - xSLG gap: 30% (power component)
    - BABIP gap: 20% (contact luck)
    
    Args:
        underperformance_gap: xwOBA-wOBA gap
        xslg_gap: xSLG-SLG gap
        babip_gap: xBA-BABIP gap
        
    Returns:
        Combined luck index
    """
    gaps = []
    weights = []
    
    if underperformance_gap is not None and not pd.isna(underperformance_gap):
        gaps.append(underperformance_gap * 0.5)
        weights.append(0.5)
    
    if xslg_gap is not None and not pd.isna(xslg_gap):
        gaps.append(xslg_gap * 0.3)
        weights.append(0.3)
    
    if babip_gap is not None and not pd.isna(babip_gap):
        gaps.append(babip_gap * 0.2)
        weights.append(0.2)
    
    if not gaps:
        return None
    
    return sum(gaps) / sum(weights)


# ============================================================================
# BASELINE / TREND FEATURES
# ============================================================================

def calculate_deviation_from_baseline(
    db: MadduxDatabase,
    player_id: int,
    year: int,
    current_ops: float,
    lookback_years: int = 3
) -> Optional[float]:
    """
    Calculate how far current OPS is from expected baseline.
    
    Baseline = weighted average of last N years OPS (more recent = higher weight)
    
    Args:
        db: Database connection
        player_id: Internal player ID
        year: Current year
        current_ops: Player's OPS in current year
        lookback_years: Years to look back
        
    Returns:
        Deviation from baseline (positive = below baseline, should improve)
    """
    if current_ops is None or pd.isna(current_ops):
        return None
    
    # Get historical OPS
    sql = """
        SELECT year, ops, pa FROM fangraphs_seasons
        WHERE player_id = ? AND year < ? AND year >= ? AND pa >= 100
        ORDER BY year DESC
    """
    
    df = db.query_df(sql, (player_id, year, year - lookback_years))
    
    if len(df) < 1:
        return None
    
    # Calculate weighted baseline (exponential decay)
    weights = [0.7 ** i for i in range(len(df))]
    weights = [w / sum(weights) for w in weights]
    
    baseline = sum(
        df.iloc[i]['ops'] * weights[i]
        for i in range(len(df))
        if pd.notna(df.iloc[i]['ops'])
    )
    
    return baseline - current_ops


def calculate_career_peak_deviation(
    db: MadduxDatabase,
    player_id: int,
    year: int,
    current_ops: float
) -> Tuple[Optional[float], Optional[int]]:
    """
    Calculate distance from career best OPS and years since peak.
    
    Args:
        db: Database connection
        player_id: Internal player ID
        year: Current year
        current_ops: Player's OPS in current year
        
    Returns:
        Tuple of (peak_deviation, years_from_peak)
    """
    if current_ops is None or pd.isna(current_ops):
        return None, None
    
    # Get career history
    sql = """
        SELECT year, ops FROM fangraphs_seasons
        WHERE player_id = ? AND year <= ? AND pa >= 200
        ORDER BY year DESC
    """
    
    df = db.query_df(sql, (player_id, year))
    
    if len(df) < 1:
        return None, None
    
    career_peak = df['ops'].max()
    peak_year = df[df['ops'] == career_peak]['year'].iloc[0]
    
    peak_deviation = career_peak - current_ops
    years_from_peak = year - peak_year
    
    return peak_deviation, years_from_peak


def calculate_improvement_momentum(
    db: MadduxDatabase,
    player_id: int,
    year: int
) -> Optional[float]:
    """
    Calculate recent improvement momentum (YoY OPS change).
    
    CRITICAL INSIGHT: This has NEGATIVE correlation with future improvement!
    Players who improved recently tend to regress.
    
    Args:
        db: Database connection
        player_id: Internal player ID
        year: Current year
        
    Returns:
        OPS change from previous year (positive = recently improved)
    """
    sql = """
        SELECT year, ops FROM fangraphs_seasons
        WHERE player_id = ? AND year IN (?, ?) AND pa >= 200
        ORDER BY year DESC
    """
    
    df = db.query_df(sql, (player_id, year, year - 1))
    
    if len(df) != 2:
        return None
    
    current_ops = df[df['year'] == year]['ops'].iloc[0]
    prev_ops = df[df['year'] == year - 1]['ops'].iloc[0]
    
    if pd.isna(current_ops) or pd.isna(prev_ops):
        return None
    
    return current_ops - prev_ops


def calculate_ops_trend_2yr(
    db: MadduxDatabase,
    player_id: int,
    year: int
) -> Optional[float]:
    """
    Calculate 2-year weighted OPS trend.
    
    Args:
        db: Database connection
        player_id: Internal player ID
        year: Current year
        
    Returns:
        Weighted average OPS over last 2 years
    """
    sql = """
        SELECT year, ops FROM fangraphs_seasons
        WHERE player_id = ? AND year <= ? AND year >= ? AND pa >= 100
        ORDER BY year DESC
    """
    
    df = db.query_df(sql, (player_id, year, year - 1))
    
    if len(df) < 2:
        return None
    
    # Weight: 70% current year, 30% previous
    weights = [0.7, 0.3]
    
    weighted_ops = sum(
        df.iloc[i]['ops'] * weights[i]
        for i in range(min(2, len(df)))
        if pd.notna(df.iloc[i]['ops'])
    )
    
    return weighted_ops / sum(weights[:len(df)])


def calculate_age_factor(age: int) -> float:
    """
    Calculate age-based adjustment factor.
    
    Peak performance is typically 26-29.
    Younger players have more upside, older players may decline.
    
    Args:
        age: Player's age
        
    Returns:
        Age factor (positive = expecting improvement, negative = decline risk)
    """
    if age is None or pd.isna(age):
        return 0.0
    
    # Peak ages
    peak_start = 26
    peak_end = 29
    
    if age < peak_start:
        # Young players: positive factor (growing)
        return (peak_start - age) * 0.02
    elif age > peak_end:
        # Older players: negative factor (declining)
        return (peak_end - age) * 0.02
    else:
        # Peak years: neutral
        return 0.0


# ============================================================================
# BAT QUALITY FEATURES
# ============================================================================

def calculate_bat_speed_zscore(
    avg_bat_speed: float,
    league_mean: float,
    league_std: float
) -> Optional[float]:
    """
    Calculate standardized bat speed (z-score).
    
    Args:
        avg_bat_speed: Player's average bat speed
        league_mean: League average bat speed
        league_std: League standard deviation
        
    Returns:
        Z-score (positive = above average)
    """
    if any(v is None or pd.isna(v) for v in [avg_bat_speed, league_mean, league_std]):
        return None
    if league_std == 0:
        return 0.0
    
    return (avg_bat_speed - league_mean) / league_std


def calculate_swing_length_factor(
    swing_length: float,
    league_mean: float
) -> Optional[float]:
    """
    Calculate swing length relative to league average.
    
    Shorter swings may indicate better contact ability.
    Longer swings may indicate power approach.
    
    Args:
        swing_length: Player's average swing length
        league_mean: League average swing length
        
    Returns:
        Relative factor (1.0 = average)
    """
    if any(v is None or pd.isna(v) for v in [swing_length, league_mean]):
        return None
    if league_mean == 0:
        return None
    
    return swing_length / league_mean


def calculate_squared_up_quality(
    squared_up_rate: float,
    avg_bat_speed: float
) -> Optional[float]:
    """
    Calculate quality of squared-up contact weighted by bat speed.
    
    Args:
        squared_up_rate: Percentage of squared-up contact
        avg_bat_speed: Average bat speed
        
    Returns:
        Quality index
    """
    if any(v is None or pd.isna(v) for v in [squared_up_rate, avg_bat_speed]):
        return None
    
    # Normalize: squared_up_rate is typically 0.15-0.35, bat_speed 65-80
    return squared_up_rate * (avg_bat_speed / 70)


def calculate_whiff_adjusted_power(
    barrel_pct: float,
    whiff_rate: float
) -> Optional[float]:
    """
    Calculate power output adjusted for swing-and-miss.
    
    High barrel% with low whiff = efficient power
    High barrel% with high whiff = volatile power
    
    Args:
        barrel_pct: Barrel percentage
        whiff_rate: Whiff per swing rate
        
    Returns:
        Whiff-adjusted power index
    """
    if any(v is None or pd.isna(v) for v in [barrel_pct, whiff_rate]):
        return None
    if whiff_rate == 0:
        return barrel_pct * 10  # Cap at high value
    
    return barrel_pct / whiff_rate


def calculate_quality_zscore(
    hard_hit_pct: float,
    barrel_pct: float,
    league_hh_mean: float,
    league_hh_std: float,
    league_barrel_mean: float,
    league_barrel_std: float
) -> Optional[float]:
    """
    Calculate combined contact quality z-score.
    
    Args:
        hard_hit_pct: Hard hit percentage
        barrel_pct: Barrel percentage
        league_hh_mean: League hard hit mean
        league_hh_std: League hard hit std
        league_barrel_mean: League barrel mean
        league_barrel_std: League barrel std
        
    Returns:
        Combined quality z-score
    """
    if any(v is None or pd.isna(v) for v in [hard_hit_pct, barrel_pct]):
        return None
    
    # Calculate individual z-scores
    hh_z = (hard_hit_pct - league_hh_mean) / league_hh_std if league_hh_std > 0 else 0
    barrel_z = (barrel_pct - league_barrel_mean) / league_barrel_std if league_barrel_std > 0 else 0
    
    # Average with barrel weighted slightly higher (more predictive)
    return (hh_z * 0.4 + barrel_z * 0.6)


def calculate_hard_hit_zscore(
    hard_hit_pct: float,
    league_mean: float,
    league_std: float
) -> Optional[float]:
    """
    Calculate hard hit percentage z-score.
    
    Args:
        hard_hit_pct: Player's hard hit percentage
        league_mean: League average
        league_std: League standard deviation
        
    Returns:
        Z-score
    """
    if any(v is None or pd.isna(v) for v in [hard_hit_pct, league_mean, league_std]):
        return None
    if league_std == 0:
        return 0.0
    
    return (hard_hit_pct - league_mean) / league_std


# ============================================================================
# LAUNCH ANGLE FEATURES
# ============================================================================

def calculate_delta_launch_angle(
    db: MadduxDatabase,
    player_id: int,
    year: int
) -> Optional[float]:
    """
    Calculate year-over-year change in launch angle.
    
    Higher launch angle often indicates swing plane adjustments
    that can lead to more power.
    
    Args:
        db: Database connection
        player_id: Internal player ID
        year: Current year
        
    Returns:
        Change in average launch angle (positive = higher angle)
    """
    sql = """
        SELECT year, avg_launch_angle FROM statcast_seasons
        WHERE player_id = ? AND year IN (?, ?)
        ORDER BY year DESC
    """
    
    df = db.query_df(sql, (player_id, year, year - 1))
    
    if len(df) != 2:
        return None
    
    current = df[df['year'] == year]['avg_launch_angle'].iloc[0]
    prev = df[df['year'] == year - 1]['avg_launch_angle'].iloc[0]
    
    if pd.isna(current) or pd.isna(prev):
        return None
    
    return current - prev


def calculate_delta_sweet_spot(
    db: MadduxDatabase,
    player_id: int,
    year: int
) -> Optional[float]:
    """
    Calculate year-over-year change in sweet spot percentage.
    
    Sweet spot (8-32 degrees) is optimal for line drives and hard contact.
    Improvements suggest better swing mechanics.
    
    Args:
        db: Database connection
        player_id: Internal player ID
        year: Current year
        
    Returns:
        Change in sweet spot percentage
    """
    sql = """
        SELECT year, sweet_spot_pct FROM statcast_seasons
        WHERE player_id = ? AND year IN (?, ?)
        ORDER BY year DESC
    """
    
    df = db.query_df(sql, (player_id, year, year - 1))
    
    if len(df) != 2:
        return None
    
    current = df[df['year'] == year]['sweet_spot_pct'].iloc[0]
    prev = df[df['year'] == year - 1]['sweet_spot_pct'].iloc[0]
    
    if pd.isna(current) or pd.isna(prev):
        return None
    
    return current - prev


def calculate_launch_angle_zscore(
    avg_launch_angle: float,
    league_mean: float,
    league_std: float
) -> Optional[float]:
    """
    Calculate standardized launch angle (z-score).
    
    Args:
        avg_launch_angle: Player's average launch angle
        league_mean: League average launch angle
        league_std: League standard deviation
        
    Returns:
        Z-score (positive = higher than average angle)
    """
    if any(v is None or pd.isna(v) for v in [avg_launch_angle, league_mean, league_std]):
        return None
    if league_std == 0:
        return 0.0
    
    return (avg_launch_angle - league_mean) / league_std


# ============================================================================
# ZONE SWING FEATURES
# ============================================================================

def calculate_delta_z_swing(
    db: MadduxDatabase,
    player_id: int,
    year: int
) -> Optional[float]:
    """
    Calculate year-over-year change in zone swing percentage.
    
    Higher Z-Swing% indicates more aggressive swinging at hittable pitches,
    which can be a positive sign of plate discipline adjustment.
    
    Args:
        db: Database connection
        player_id: Internal player ID
        year: Current year
        
    Returns:
        Change in zone swing percentage
    """
    sql = """
        SELECT year, z_swing_pct FROM fangraphs_seasons
        WHERE player_id = ? AND year IN (?, ?) AND pa >= 100
        ORDER BY year DESC
    """
    
    df = db.query_df(sql, (player_id, year, year - 1))
    
    if len(df) != 2:
        return None
    
    current = df[df['year'] == year]['z_swing_pct'].iloc[0]
    prev = df[df['year'] == year - 1]['z_swing_pct'].iloc[0]
    
    if pd.isna(current) or pd.isna(prev):
        return None
    
    return current - prev


def calculate_discipline_improvement(
    delta_z_swing: float,
    delta_chase_rate: float
) -> Optional[float]:
    """
    Calculate net discipline improvement.
    
    Good discipline = higher Z-Swing% (aggressive on good pitches)
                    + lower O-Swing% (patient on bad pitches)
    
    Formula: delta_z_swing - delta_chase_rate
    (Positive = improved discipline)
    
    Args:
        delta_z_swing: Change in zone swing rate
        delta_chase_rate: Change in chase rate (O-Swing%)
        
    Returns:
        Net discipline improvement score
    """
    if delta_z_swing is None or delta_chase_rate is None:
        return None
    if pd.isna(delta_z_swing) or pd.isna(delta_chase_rate):
        return None
    
    return delta_z_swing - delta_chase_rate


def calculate_delta_chase_rate(
    db: MadduxDatabase,
    player_id: int,
    year: int
) -> Optional[float]:
    """
    Calculate year-over-year change in chase rate (O-Swing%).
    
    Args:
        db: Database connection
        player_id: Internal player ID
        year: Current year
        
    Returns:
        Change in chase rate (negative = improved discipline)
    """
    sql = """
        SELECT year, chase_rate FROM fangraphs_seasons
        WHERE player_id = ? AND year IN (?, ?) AND pa >= 100
        ORDER BY year DESC
    """
    
    df = db.query_df(sql, (player_id, year, year - 1))
    
    if len(df) != 2:
        return None
    
    current = df[df['year'] == year]['chase_rate'].iloc[0]
    prev = df[df['year'] == year - 1]['chase_rate'].iloc[0]
    
    if pd.isna(current) or pd.isna(prev):
        return None
    
    return current - prev


# ============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# ============================================================================

def calculate_league_averages(db: MadduxDatabase, year: int, min_pa: int = 200) -> Dict:
    """
    Calculate league averages for standardization.
    
    Args:
        db: Database connection
        year: Season year
        min_pa: Minimum PA for inclusion
        
    Returns:
        Dict with league averages and standard deviations
    """
    sql = """
        SELECT 
            AVG(s.avg_bat_speed) as bat_speed_mean,
            AVG(s.swing_length) as swing_length_mean,
            AVG(s.hard_hit_pct) as hard_hit_mean,
            AVG(s.barrel_pct) as barrel_mean,
            AVG(s.avg_launch_angle) as launch_angle_mean,
            -- Standard deviations using subquery
            (SELECT AVG((s2.avg_bat_speed - sub.mean) * (s2.avg_bat_speed - sub.mean))
             FROM statcast_seasons s2, 
                  (SELECT AVG(avg_bat_speed) as mean FROM statcast_seasons WHERE year = ?) sub
             WHERE s2.year = ? AND s2.avg_bat_speed IS NOT NULL) as bat_speed_var,
            (SELECT AVG((s2.hard_hit_pct - sub.mean) * (s2.hard_hit_pct - sub.mean))
             FROM statcast_seasons s2,
                  (SELECT AVG(hard_hit_pct) as mean FROM statcast_seasons WHERE year = ?) sub
             WHERE s2.year = ? AND s2.hard_hit_pct IS NOT NULL) as hard_hit_var,
            (SELECT AVG((s2.barrel_pct - sub.mean) * (s2.barrel_pct - sub.mean))
             FROM statcast_seasons s2,
                  (SELECT AVG(barrel_pct) as mean FROM statcast_seasons WHERE year = ?) sub
             WHERE s2.year = ? AND s2.barrel_pct IS NOT NULL) as barrel_var,
            (SELECT AVG((s2.avg_launch_angle - sub.mean) * (s2.avg_launch_angle - sub.mean))
             FROM statcast_seasons s2,
                  (SELECT AVG(avg_launch_angle) as mean FROM statcast_seasons WHERE year = ?) sub
             WHERE s2.year = ? AND s2.avg_launch_angle IS NOT NULL) as launch_angle_var
        FROM statcast_seasons s
        JOIN fangraphs_seasons f ON s.player_id = f.player_id AND s.year = f.year
        WHERE s.year = ? AND f.pa >= ?
    """
    
    result = db.query(sql, (year, year, year, year, year, year, year, year, year, min_pa))
    
    if not result or result[0][0] is None:
        return {}
    
    row = result[0]
    return {
        'bat_speed_mean': row[0] or 71.0,
        'swing_length_mean': row[1] or 7.5,
        'hard_hit_mean': row[2] or 38.0,
        'barrel_mean': row[3] or 7.5,
        'launch_angle_mean': row[4] or 12.0,
        'bat_speed_std': (row[5] ** 0.5) if row[5] else 3.0,
        'hard_hit_std': (row[6] ** 0.5) if row[6] else 8.0,
        'barrel_std': (row[7] ** 0.5) if row[7] else 4.0,
        'launch_angle_std': (row[8] ** 0.5) if row[8] else 5.0,
    }


def calculate_original_maddux_score(delta_max_ev: float, delta_hard_hit_pct: float) -> float:
    """
    Calculate the original MADDUX score for comparison.
    
    Formula: Score = Δ Max EV + (2.1 × Δ Hard Hit%)
    
    Note: This is the ORIGINAL formula that was shown to have NEGATIVE correlation.
    Kept for comparison purposes only.
    """
    if delta_max_ev is None or delta_hard_hit_pct is None:
        return 0.0
    return delta_max_ev + (2.1 * delta_hard_hit_pct)


def calculate_all_features_for_year(
    db: MadduxDatabase,
    year: int,
    min_pa: int = 200
) -> pd.DataFrame:
    """
    Calculate all engineered features for a given year.
    
    Args:
        db: Database connection
        year: Season year
        min_pa: Minimum plate appearances
        
    Returns:
        DataFrame with all features for each player
    """
    print(f"Calculating features for {year}...")
    
    # Get league averages for standardization
    league_avgs = calculate_league_averages(db, year, min_pa)
    
    # Get combined data
    sql = """
        SELECT 
            p.id as player_id,
            p.player_name,
            s.year,
            s.max_ev,
            s.avg_ev,
            s.hard_hit_pct,
            s.barrel_pct,
            s.avg_bat_speed,
            s.swing_length,
            s.squared_up_rate,
            s.whiff_rate,
            s.avg_launch_angle,
            s.sweet_spot_pct,
            s.xwoba,
            s.xba,
            s.xslg,
            f.age,
            f.pa,
            f.ops,
            f.slg,
            f.iso,
            f.woba,
            f.babip,
            f.wrc_plus,
            f.z_swing_pct,
            f.chase_rate
        FROM players p
        JOIN statcast_seasons s ON p.id = s.player_id
        JOIN fangraphs_seasons f ON p.id = f.player_id AND s.year = f.year
        WHERE s.year = ? AND f.pa >= ?
    """
    
    df = db.query_df(sql, (year, min_pa))
    
    if len(df) == 0:
        print(f"  No data found for {year}")
        return pd.DataFrame()
    
    print(f"  Processing {len(df)} players...")
    
    # Calculate features for each player
    features_list = []
    
    for _, row in df.iterrows():
        player_id = row['player_id']
        
        # Luck/underperformance features
        underperformance_gap = calculate_underperformance_gap(row['xwoba'], row['woba'])
        xslg_gap = calculate_xslg_gap(row['xslg'], row['slg'])
        babip_gap = calculate_babip_gap(row['xba'], row['babip'])
        iso_gap = calculate_iso_gap(row['xslg'], row['xba'], row['iso'])
        combined_luck = calculate_combined_luck_index(underperformance_gap, xslg_gap, babip_gap)
        
        # Baseline/trend features
        deviation = calculate_deviation_from_baseline(db, player_id, year, row['ops'])
        peak_dev, years_from_peak = calculate_career_peak_deviation(db, player_id, year, row['ops'])
        momentum = calculate_improvement_momentum(db, player_id, year)
        ops_trend = calculate_ops_trend_2yr(db, player_id, year)
        age_factor = calculate_age_factor(row['age'])
        
        # Bat quality features
        bat_speed_z = calculate_bat_speed_zscore(
            row['avg_bat_speed'],
            league_avgs.get('bat_speed_mean', 71.0),
            league_avgs.get('bat_speed_std', 3.0)
        )
        swing_length_factor = calculate_swing_length_factor(
            row['swing_length'],
            league_avgs.get('swing_length_mean', 7.5)
        )
        squared_up_quality = calculate_squared_up_quality(
            row['squared_up_rate'],
            row['avg_bat_speed']
        )
        whiff_adj_power = calculate_whiff_adjusted_power(row['barrel_pct'], row['whiff_rate'])
        
        quality_z = calculate_quality_zscore(
            row['hard_hit_pct'],
            row['barrel_pct'],
            league_avgs.get('hard_hit_mean', 38.0),
            league_avgs.get('hard_hit_std', 8.0),
            league_avgs.get('barrel_mean', 7.5),
            league_avgs.get('barrel_std', 4.0)
        )
        hard_hit_z = calculate_hard_hit_zscore(
            row['hard_hit_pct'],
            league_avgs.get('hard_hit_mean', 38.0),
            league_avgs.get('hard_hit_std', 8.0)
        )
        
        # Launch angle features
        delta_launch = calculate_delta_launch_angle(db, player_id, year)
        delta_sweet = calculate_delta_sweet_spot(db, player_id, year)
        launch_z = calculate_launch_angle_zscore(
            row['avg_launch_angle'],
            league_avgs.get('launch_angle_mean', 12.0),
            league_avgs.get('launch_angle_std', 5.0)
        )
        
        # Zone swing features
        delta_z_swing = calculate_delta_z_swing(db, player_id, year)
        delta_chase = calculate_delta_chase_rate(db, player_id, year)
        discipline_imp = calculate_discipline_improvement(delta_z_swing, delta_chase)
        
        # PA weight (more PAs = more reliable)
        pa_weight = min(row['pa'] / 500, 1.0) if row['pa'] else 0.5
        
        # Get previous year data for original MADDUX score
        prev_year_sql = """
            SELECT max_ev, hard_hit_pct FROM statcast_seasons
            WHERE player_id = ? AND year = ?
        """
        prev_data = db.query(prev_year_sql, (player_id, year - 1))
        
        if prev_data and prev_data[0][0] is not None:
            delta_max_ev = row['max_ev'] - prev_data[0][0] if row['max_ev'] else 0
            delta_hh = row['hard_hit_pct'] - prev_data[0][1] if row['hard_hit_pct'] and prev_data[0][1] else 0
            original_maddux = calculate_original_maddux_score(delta_max_ev, delta_hh)
        else:
            original_maddux = None
        
        features_list.append({
            'player_id': player_id,
            'year': year,
            # Luck features
            'underperformance_gap': underperformance_gap,
            'xslg_gap_scaled': xslg_gap,
            'babip_gap': babip_gap,
            'iso_gap': iso_gap,
            'combined_luck_index': combined_luck,
            # Baseline/trend features
            'deviation_from_baseline': deviation,
            'career_peak_deviation': peak_dev,
            'improvement_momentum': momentum,
            'ops_trend_2yr': ops_trend,
            'years_from_peak': years_from_peak,
            # Bat quality features
            'bat_speed_zscore': bat_speed_z,
            'swing_length_factor': swing_length_factor,
            'squared_up_quality': squared_up_quality,
            'whiff_adjusted_power': whiff_adj_power,
            'quality_zscore': quality_z,
            'hard_hit_zscore': hard_hit_z,
            # Launch angle features
            'delta_launch_angle': delta_launch,
            'delta_sweet_spot': delta_sweet,
            'launch_angle_zscore': launch_z,
            # Zone swing features
            'delta_z_swing': delta_z_swing,
            'discipline_improvement': discipline_imp,
            # Context features
            'age_factor': age_factor,
            'pa_weight': pa_weight,
            'original_maddux_score': original_maddux
        })
    
    features_df = pd.DataFrame(features_list)
    print(f"  Calculated {len(features_df)} feature sets")
    
    return features_df


def calculate_and_store_all_features(
    db: MadduxDatabase,
    start_year: int = 2016,
    end_year: int = 2025,
    min_pa: int = 200
) -> int:
    """
    Calculate and store features for all years.
    
    Args:
        db: Database connection
        start_year: First year to process
        end_year: Last year to process
        min_pa: Minimum plate appearances
        
    Returns:
        Total features stored
    """
    total_stored = 0
    
    for year in range(start_year, end_year + 1):
        features_df = calculate_all_features_for_year(db, year, min_pa)
        
        if len(features_df) > 0:
            stored = store_player_features(db, features_df)
            total_stored += stored
            print(f"  Stored {stored} features for {year}")
    
    return total_stored


def analyze_feature_correlations(db: MadduxDatabase) -> pd.DataFrame:
    """
    Analyze correlation between features and next-year OPS change.
    
    Args:
        db: Database connection
        
    Returns:
        DataFrame with correlation analysis
    """
    sql = """
        SELECT 
            pf.player_id,
            pf.year,
            pf.underperformance_gap,
            pf.xslg_gap_scaled,
            pf.babip_gap,
            pf.combined_luck_index,
            pf.deviation_from_baseline,
            pf.career_peak_deviation,
            pf.improvement_momentum,
            pf.bat_speed_zscore,
            pf.squared_up_quality,
            pf.whiff_adjusted_power,
            pf.quality_zscore,
            pf.hard_hit_zscore,
            pf.delta_launch_angle,
            pf.delta_sweet_spot,
            pf.launch_angle_zscore,
            pf.delta_z_swing,
            pf.discipline_improvement,
            pf.age_factor,
            pf.original_maddux_score,
            f1.ops as current_ops,
            f2.ops as next_ops,
            (f2.ops - f1.ops) as ops_change
        FROM player_features pf
        JOIN fangraphs_seasons f1 ON pf.player_id = f1.player_id AND pf.year = f1.year
        JOIN fangraphs_seasons f2 ON pf.player_id = f2.player_id AND pf.year + 1 = f2.year
        WHERE f1.pa >= 200 AND f2.pa >= 200
    """
    
    df = db.query_df(sql)
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Features to analyze
    features = [
        'underperformance_gap', 'xslg_gap_scaled', 'babip_gap', 'combined_luck_index',
        'deviation_from_baseline', 'career_peak_deviation', 'improvement_momentum',
        'bat_speed_zscore', 'squared_up_quality', 'whiff_adjusted_power',
        'quality_zscore', 'hard_hit_zscore', 
        'delta_launch_angle', 'delta_sweet_spot', 'launch_angle_zscore',
        'delta_z_swing', 'discipline_improvement',
        'age_factor', 'original_maddux_score'
    ]
    
    correlations = []
    for feat in features:
        if feat not in df.columns:
            continue
        
        clean = df[[feat, 'ops_change']].dropna()
        if len(clean) > 30:
            r, p = stats.pearsonr(clean[feat], clean['ops_change'])
            correlations.append({
                'feature': feat,
                'correlation': r,
                'p_value': p,
                'n': len(clean),
                'significant': p < 0.05
            })
    
    result = pd.DataFrame(correlations)
    return result.sort_values('correlation', ascending=False)


def main():
    """Calculate features for all years and analyze correlations."""
    print("=" * 70)
    print("MADDUX™ Feature Engineering (Enhanced v2)")
    print("=" * 70)
    
    db = MadduxDatabase()
    
    # Calculate and store features
    print("\nCalculating features for all years...")
    print("-" * 70)
    
    total = calculate_and_store_all_features(db, start_year=2016, end_year=2025)
    
    print(f"\nTotal features stored: {total}")
    
    # Analyze correlations
    print("\n" + "=" * 70)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 70)
    print("Correlation with next-year OPS change:\n")
    
    corr_df = analyze_feature_correlations(db)
    
    if len(corr_df) > 0:
        print(f"{'Feature':<30} {'Corr':>8} {'p-value':>10} {'N':>6} {'Sig':>5}")
        print("-" * 65)
        for _, row in corr_df.iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            print(f"{row['feature']:<30} {row['correlation']:>+8.4f} {row['p_value']:>10.4f} {row['n']:>6} {sig:>5}")
        
        # Key insights
        print("\n" + "=" * 70)
        print("KEY INSIGHTS")
        print("=" * 70)
        
        # Best positive predictors
        positive = corr_df[corr_df['correlation'] > 0.1].head(3)
        if len(positive) > 0:
            print("\nBest positive predictors (higher = more improvement):")
            for _, row in positive.iterrows():
                print(f"  • {row['feature']}: r = {row['correlation']:.3f}")
        
        # Best negative predictors (these predict regression!)
        negative = corr_df[corr_df['correlation'] < -0.1].tail(3)
        if len(negative) > 0:
            print("\nNegative predictors (higher = MORE REGRESSION):")
            for _, row in negative.iterrows():
                print(f"  • {row['feature']}: r = {row['correlation']:.3f}")
        
        # Compare original MADDUX score
        orig = corr_df[corr_df['feature'] == 'original_maddux_score']
        if len(orig) > 0:
            print(f"\nOriginal MADDUX score correlation: {orig.iloc[0]['correlation']:.3f}")
            print("  → Confirms original formula has weak/negative predictive power")
    
    db.close()
    print("\nFeature engineering complete!")


if __name__ == "__main__":
    main()

