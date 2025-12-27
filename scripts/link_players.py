"""
link_players.py
Links player records between Statcast and FanGraphs data sources.

Uses pybaseball to map MLB IDs to FanGraphs IDs and merges player records.
"""

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import MadduxDatabase

# Try pybaseball for ID mapping
try:
    from pybaseball import playerid_reverse_lookup
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False


def get_unlinked_players(db: MadduxDatabase) -> Dict[str, List]:
    """Get players that need linking."""
    # Players with MLB ID but no FanGraphs ID
    mlb_only = db.query("""
        SELECT id, mlb_id, player_name 
        FROM players 
        WHERE mlb_id IS NOT NULL AND fangraphs_id IS NULL
    """)
    
    # Players with FanGraphs ID but no MLB ID
    fg_only = db.query("""
        SELECT id, fangraphs_id, player_name 
        FROM players 
        WHERE fangraphs_id IS NOT NULL AND mlb_id IS NULL
    """)
    
    return {
        'mlb_only': mlb_only,
        'fg_only': fg_only
    }


def link_via_pybaseball(db: MadduxDatabase) -> int:
    """
    Link players using pybaseball ID lookup.
    
    Returns number of players linked.
    """
    if not PYBASEBALL_AVAILABLE:
        print("pybaseball not available")
        return 0
    
    # Get MLB IDs that need FanGraphs IDs
    mlb_ids = db.query("""
        SELECT id, mlb_id, player_name
        FROM players
        WHERE mlb_id IS NOT NULL AND fangraphs_id IS NULL
    """)
    
    if not mlb_ids:
        return 0
    
    mlb_id_list = [row[1] for row in mlb_ids]
    id_map = {row[1]: row[0] for row in mlb_ids}  # mlb_id -> internal_id
    
    try:
        # Look up FanGraphs IDs
        print(f"Looking up {len(mlb_id_list)} MLB IDs...")
        mapping = playerid_reverse_lookup(mlb_id_list, key_type='mlbam')
        
        linked = 0
        for _, row in mapping.iterrows():
            mlb_id = row.get('key_mlbam')
            fg_id = row.get('key_fangraphs')
            
            if pd.notna(mlb_id) and pd.notna(fg_id):
                internal_id = id_map.get(int(mlb_id))
                if internal_id:
                    # Check if there's already a player with this FanGraphs ID
                    existing = db.query(
                        "SELECT id FROM players WHERE fangraphs_id = ?",
                        (int(fg_id),)
                    )
                    
                    if existing:
                        # Merge: move FanGraphs seasons to MLB player, delete FG-only player
                        fg_player_id = existing[0][0]
                        
                        # Update FanGraphs seasons to point to MLB player
                        db.execute("""
                            UPDATE fangraphs_seasons 
                            SET player_id = ? 
                            WHERE player_id = ?
                        """, (internal_id, fg_player_id))
                        
                        # Update the MLB player with FanGraphs ID
                        db.execute(
                            "UPDATE players SET fangraphs_id = ? WHERE id = ?",
                            (int(fg_id), internal_id)
                        )
                        
                        # Delete the duplicate FG-only player
                        db.execute("DELETE FROM players WHERE id = ?", (fg_player_id,))
                        linked += 1
                    else:
                        # Just update the FanGraphs ID
                        db.execute(
                            "UPDATE players SET fangraphs_id = ? WHERE id = ?",
                            (int(fg_id), internal_id)
                        )
                        linked += 1
        
        db.commit()
        return linked
        
    except Exception as e:
        print(f"Error in pybaseball lookup: {e}")
        return 0


def link_via_name_matching(db: MadduxDatabase) -> int:
    """
    Link players by matching names between sources.
    
    This is a fallback when ID lookup fails.
    """
    # Get unlinked FanGraphs players
    fg_players = db.query("""
        SELECT id, fangraphs_id, player_name
        FROM players
        WHERE fangraphs_id IS NOT NULL AND mlb_id IS NULL
    """)
    
    if not fg_players:
        return 0
    
    linked = 0
    for fg_id, fangraphs_id, fg_name in fg_players:
        # Try to find matching MLB player by name
        # Normalize name for comparison
        normalized_name = fg_name.lower().strip()
        
        mlb_match = db.query("""
            SELECT id, mlb_id, player_name
            FROM players
            WHERE mlb_id IS NOT NULL 
              AND fangraphs_id IS NULL
              AND LOWER(TRIM(player_name)) = ?
        """, (normalized_name,))
        
        if mlb_match:
            mlb_internal_id = mlb_match[0][0]
            
            # Move FanGraphs seasons to MLB player
            db.execute("""
                UPDATE fangraphs_seasons 
                SET player_id = ? 
                WHERE player_id = ?
            """, (mlb_internal_id, fg_id))
            
            # Update MLB player with FanGraphs ID
            db.execute("""
                UPDATE players 
                SET fangraphs_id = ? 
                WHERE id = ?
            """, (fangraphs_id, mlb_internal_id))
            
            # Delete duplicate FG player
            db.execute("DELETE FROM players WHERE id = ?", (fg_id,))
            linked += 1
    
    db.commit()
    return linked


def verify_linking(db: MadduxDatabase) -> Dict:
    """Verify player linking status."""
    stats = {}
    
    # Total players
    stats['total_players'] = db.query("SELECT COUNT(*) FROM players")[0][0]
    
    # Players with both IDs
    stats['fully_linked'] = db.query("""
        SELECT COUNT(*) FROM players 
        WHERE mlb_id IS NOT NULL AND fangraphs_id IS NOT NULL
    """)[0][0]
    
    # Players with only MLB ID
    stats['mlb_only'] = db.query("""
        SELECT COUNT(*) FROM players 
        WHERE mlb_id IS NOT NULL AND fangraphs_id IS NULL
    """)[0][0]
    
    # Players with only FanGraphs ID
    stats['fg_only'] = db.query("""
        SELECT COUNT(*) FROM players 
        WHERE fangraphs_id IS NOT NULL AND mlb_id IS NULL
    """)[0][0]
    
    # Statcast seasons with FanGraphs data
    stats['linked_statcast_seasons'] = db.query("""
        SELECT COUNT(*)
        FROM statcast_seasons s
        JOIN players p ON s.player_id = p.id
        JOIN fangraphs_seasons f ON p.id = f.player_id AND s.year = f.year
    """)[0][0]
    
    return stats


def main():
    """Run player linking."""
    print("=" * 60)
    print("MADDUX Phase 1 - Player Linking")
    print("=" * 60)
    
    db = MadduxDatabase()
    
    # Initial status
    print("\nBefore linking:")
    stats = verify_linking(db)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Link via pybaseball
    print("\nLinking via pybaseball ID lookup...")
    linked_pybaseball = link_via_pybaseball(db)
    print(f"  Linked: {linked_pybaseball}")
    
    # Link via name matching
    print("\nLinking via name matching...")
    linked_names = link_via_name_matching(db)
    print(f"  Linked: {linked_names}")
    
    # Final status
    print("\nAfter linking:")
    stats = verify_linking(db)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    db.close()
    print("\nPlayer linking complete!")


if __name__ == "__main__":
    main()

