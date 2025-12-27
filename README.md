# MADDUX™ Hitter Analytics - Phase 1

**Predictive Baseball Analytics for Hitter Breakouts**  
*Powered by AutoCoach LLC*  
*December 2024*

---

## Overview

MADDUX™ is a predictive analytics system that identifies hitter breakout candidates using advanced baseball metrics from Baseball Savant (Statcast) and FanGraphs.

### Phase 1 Deliverables

| Requirement | Status | Location |
|-------------|--------|----------|
| Data Pull (2015-2025) | ✅ Complete | `data/` |
| SQLite Database | ✅ Complete | `database/maddux.db` |
| Claude API Integration | ✅ Complete | `scripts/claude_query.py` |
| Visualization Dashboard | ✅ Complete | `dashboard/index.html` |
| Interactive Features | ✅ Complete | Multi-tab dashboard |

### Model Performance

| Metric | Result |
|--------|--------|
| Walk-Forward Hit Rate | **79.3%** |
| Correlation (r) | **0.50** |
| R-squared | **28%** |
| Years Validated | 7 (2018-2025) |
| Player-Seasons | ~1,700 |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. View the Dashboard

Open `dashboard/index.html` in any web browser to see:
- Model overview and key metrics
- 2026 breakout projections
- Historical validation results
- Top performer analysis

### 3. Query with Claude (Optional)

```bash
export ANTHROPIC_API_KEY="your-api-key"
python scripts/claude_query.py
```

### 4. Run the Full Pipeline

```bash
# Pull fresh data (if needed)
python scripts/pull_multi_year.py

# Initialize database
python scripts/database.py

# Calculate features
python scripts/feature_engineering.py

# Run validation
python scripts/backtest.py

# Generate projections
python scripts/projections.py
```

---

## Project Structure

```
maddux_phase_1/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── data/                     # Raw data (2015-2025)
│   ├── 2015/ ... 2025/      # Year-by-year CSVs
│   │   ├── statcast.csv     # Exit velocity, barrels, xStats
│   │   └── fangraphs.csv    # OPS, wRC+, PA, batting stats
│   ├── all_statcast.csv     # Combined Statcast data
│   └── all_fangraphs.csv    # Combined FanGraphs data
│
├── database/
│   └── maddux.db            # SQLite database
│
├── scripts/
│   ├── database.py          # Database schema & loading
│   ├── pull_multi_year.py   # Data pulling from sources
│   ├── feature_engineering.py # Engineered feature calculations
│   ├── stacking_model.py    # Ensemble model
│   ├── maddux_algorithm.py  # Core algorithm
│   ├── regression.py        # Statistical validation
│   ├── backtest.py          # Walk-forward validation
│   ├── projections.py       # 2026 predictions
│   ├── claude_query.py      # Claude API integration
│   └── export_sheets.py     # CSV exports
│
├── dashboard/
│   ├── index.html           # Interactive dashboard
│   └── data/                # Dashboard JSON data
│
├── sheets_output/           # CSV exports for analysis
│
├── tests/                   # Test suite
│
└── docs/
    ├── VALIDATION_MEMO.md   # Model validation report
    └── SERVICE_DESIGN.md    # Phase 2 architecture
```

---

## Model Approach

The model uses **regression-to-mean principles** to identify underperforming hitters likely to improve:

### Key Predictive Features

| Feature | Correlation | Description |
|---------|-------------|-------------|
| deviation_from_baseline | +0.50 | Distance from expected OPS |
| career_peak_deviation | +0.38 | Distance from career best |
| underperformance_gap | +0.35 | xwOBA minus actual wOBA |
| age_factor | +0.30 | Age-based adjustment |

### Key Insight

Players who recently improved are likely to **regress** next year. The model identifies players whose current performance is *below* their expected level.

---

## Top 2026 Breakout Candidates

| Rank | Player | Age | 2025 OPS | Predicted Δ | 2026 OPS |
|------|--------|-----|----------|-------------|----------|
| 1 | LaMonte Wade Jr. | 31 | .524 | +.121 | .645 |
| 2 | Joc Pederson | 33 | .614 | +.114 | .728 |
| 3 | Henry Davis | 25 | .512 | +.105 | .617 |
| 4 | Anthony Santander | 30 | .565 | +.092 | .657 |
| 5 | Tyler O'Neill | 30 | .684 | +.091 | .775 |
| 6 | Jordan Walker | 23 | .584 | +.089 | .673 |
| 7 | Matt McLain | 25 | .643 | +.079 | .722 |
| 8 | Mookie Betts | 32 | .732 | +.063 | .795 |
| 9 | Oneil Cruz | 26 | .676 | +.053 | .729 |
| 10 | Luis Robert Jr. | 27 | .661 | +.053 | .714 |

Full projections available in `dashboard/index.html` and `sheets_output/5_projections_2026.csv`.

---

## Data Sources

- **Baseball Savant (Statcast):** Exit velocity, hard hit %, barrels, xwOBA, xSLG
- **FanGraphs:** OPS, wRC+, PA, wOBA, BABIP, age

Data covers 2015-2025 seasons with ~1,700 qualified hitter-seasons (300+ PA).

---

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- pybaseball
- anthropic (for Claude integration)

See `requirements.txt` for full list.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Phase 2 Preview

The validated model is ready for Phase 2 real-time tracking:

- Rolling in-season calculations
- Alert system for threshold crossings
- Live dashboard updates
- API endpoints

See `docs/SERVICE_DESIGN.md` for architecture details.

---

*Built with pybaseball, scikit-learn, and Claude AI*  
*© 2024 AutoCoach LLC*
