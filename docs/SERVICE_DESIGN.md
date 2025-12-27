# MADDUX™ Real-Time Tracking Service Design

**Version:** 1.0  
**Date:** December 2025  
**Status:** Phase 2 Architecture Proposal

---

## Overview

This document outlines the architecture for the MADDUX™ real-time tracking service, designed to monitor player metric changes throughout the 2026 season and identify breakout candidates in near real-time.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MADDUX™ Service Architecture              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Baseball   │    │   FanGraphs  │    │   Manual     │  │
│  │    Savant    │    │     API      │    │   Upload     │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                    │          │
│         └─────────┬─────────┴────────────────────┘          │
│                   ▼                                          │
│         ┌─────────────────┐                                 │
│         │  Data Ingestion │                                 │
│         │     Service     │                                 │
│         └────────┬────────┘                                 │
│                  ▼                                          │
│         ┌─────────────────┐                                 │
│         │   PostgreSQL    │                                 │
│         │    Database     │                                 │
│         └────────┬────────┘                                 │
│                  │                                          │
│    ┌─────────────┼─────────────┐                           │
│    ▼             ▼             ▼                           │
│ ┌──────┐   ┌──────────┐   ┌──────────┐                    │
│ │ REST │   │  Score   │   │  Alert   │                    │
│ │ API  │   │ Calculator│   │ Service  │                    │
│ └──┬───┘   └──────────┘   └────┬─────┘                    │
│    │                           │                           │
│    ▼                           ▼                           │
│ ┌──────────────┐       ┌──────────────┐                   │
│ │  Dashboard   │       │    Email/    │                   │
│ │     UI       │       │   Webhook    │                   │
│ └──────────────┘       └──────────────┘                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### Core Tables

```sql
-- Players master table
CREATE TABLE players (
    id SERIAL PRIMARY KEY,
    mlb_id INTEGER UNIQUE NOT NULL,
    fangraphs_id INTEGER,
    player_name VARCHAR(100) NOT NULL,
    team VARCHAR(50),
    position VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Statcast metrics (historical + current)
CREATE TABLE statcast_metrics (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    date DATE NOT NULL,
    max_ev DECIMAL(5,1),
    avg_ev DECIMAL(5,1),
    hard_hit_pct DECIMAL(5,2),
    barrel_pct DECIMAL(5,2),
    avg_bat_speed DECIMAL(5,2),
    batted_balls INTEGER,
    UNIQUE(player_id, date)
);

-- Rolling calculations
CREATE TABLE rolling_metrics (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    calculation_date DATE NOT NULL,
    window_days INTEGER NOT NULL,  -- 7, 30, 90, season
    max_ev DECIMAL(5,1),
    hard_hit_pct DECIMAL(5,2),
    delta_max_ev DECIMAL(5,2),  -- vs previous window
    delta_hard_hit DECIMAL(5,2),
    maddux_score DECIMAL(6,2),
    UNIQUE(player_id, calculation_date, window_days)
);

-- Performance tracking
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    date DATE NOT NULL,
    pa INTEGER,
    ops DECIMAL(4,3),
    wrc_plus INTEGER,
    UNIQUE(player_id, date)
);

-- Alert history
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    alert_type VARCHAR(50),
    maddux_score DECIMAL(6,2),
    threshold VARCHAR(20),
    triggered_at TIMESTAMP DEFAULT NOW(),
    notified BOOLEAN DEFAULT FALSE
);
```

### Indexes

```sql
CREATE INDEX idx_statcast_player_date ON statcast_metrics(player_id, date);
CREATE INDEX idx_rolling_player_date ON rolling_metrics(player_id, calculation_date);
CREATE INDEX idx_rolling_score ON rolling_metrics(maddux_score DESC);
CREATE INDEX idx_alerts_unnotified ON alerts(notified) WHERE notified = FALSE;
```

---

## REST API Endpoints

### Players

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/players` | List all players (with pagination) |
| GET | `/api/players/{id}` | Get player details |
| GET | `/api/players/{id}/metrics` | Get player metric history |
| GET | `/api/players/{id}/scores` | Get MADDUX score history |
| GET | `/api/players/search?q=` | Search players by name |

### Rankings

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/rankings` | Current MADDUX rankings |
| GET | `/api/rankings/history` | Historical ranking snapshots |
| GET | `/api/rankings/risers` | Players with biggest score increases |
| GET | `/api/rankings/fallers` | Players with biggest score decreases |

### Projections

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/projections` | Current breakout candidates |
| GET | `/api/projections/{year}` | Projections for specific year |
| POST | `/api/projections/calculate` | Recalculate with custom weights |

### Alerts

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/alerts` | Get recent alerts |
| POST | `/api/alerts/subscribe` | Subscribe to alerts |
| DELETE | `/api/alerts/unsubscribe` | Unsubscribe from alerts |

### Example Response

```json
// GET /api/players/123/scores
{
  "player_id": 123,
  "player_name": "Brice Turang",
  "team": "MIL",
  "current_score": {
    "date": "2025-06-15",
    "maddux_score": 28.5,
    "delta_max_ev": 2.1,
    "delta_hard_hit": 12.5,
    "window": "30_day"
  },
  "score_history": [
    {"date": "2025-06-01", "score": 22.3},
    {"date": "2025-05-15", "score": 18.7}
  ],
  "rank": 5,
  "rank_change": "+3"
}
```

---

## Rolling Calculation Logic

### Update Schedule

| Frequency | Calculation | Purpose |
|-----------|-------------|---------|
| Daily | 7-day rolling metrics | Identify hot streaks |
| Weekly | 30-day rolling + deltas | Main tracking window |
| Monthly | 90-day + season-to-date | Trend confirmation |

### Score Calculation

```python
def calculate_rolling_score(player_id, window_days=30, weight=2.1):
    """
    Calculate MADDUX score based on rolling window.
    
    1. Get metrics for current window
    2. Get metrics for previous window
    3. Calculate deltas
    4. Apply MADDUX formula
    """
    
    current = get_metrics(player_id, window_days, offset=0)
    previous = get_metrics(player_id, window_days, offset=window_days)
    
    delta_max_ev = current.max_ev - previous.max_ev
    delta_hard_hit = current.hard_hit_pct - previous.hard_hit_pct
    
    score = delta_max_ev + (weight * delta_hard_hit)
    
    return {
        'score': score,
        'delta_max_ev': delta_max_ev,
        'delta_hard_hit': delta_hard_hit
    }
```

---

## Alert Threshold Definitions

### Alert Tiers

| Tier | Criteria | Action |
|------|----------|--------|
| **Critical** | Score > 25 + both metrics improving | Immediate email + webhook |
| **High** | Score > 20 OR score crossed 15 threshold | Daily digest |
| **Medium** | Score > 15 for first time | Weekly summary |
| **Watch** | Score 10-15, trending up | Dashboard highlight |

### Alert Configuration

```json
{
  "alert_config": {
    "critical": {
      "score_threshold": 25,
      "require_both_improving": true,
      "notify_immediately": true,
      "channels": ["email", "webhook", "sms"]
    },
    "high": {
      "score_threshold": 20,
      "threshold_crossing": 15,
      "notify_daily": true,
      "channels": ["email"]
    },
    "medium": {
      "score_threshold": 15,
      "first_occurrence": true,
      "notify_weekly": true,
      "channels": ["email"]
    }
  }
}
```

---

## Dashboard UI Wireframe

```
┌────────────────────────────────────────────────────────────────┐
│  MADDUX™ Tracker                    🔔 3 alerts    [Settings]  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─── Quick Stats ──────────────────────────────────────────┐ │
│  │ 📈 12 Risers    📉 8 Fallers    ⚠️ 3 Alerts    👁️ 45 Watch│ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌─── Top Candidates (30-Day) ──────────────────────────────┐ │
│  │ Rank  Player           Score   Δ MaxEV   Δ HH%   Trend   │ │
│  │  1    B. Turang        28.5    +2.1      +12.5    ↑↑     │ │
│  │  2    N. Fortes        25.3    +1.8      +11.2    ↑      │ │
│  │  3    B. Baty          24.1    +3.2      +10.0    ↑↑↑    │ │
│  │  4    M. Vargas        22.8    +0.5      +10.6    →      │ │
│  │  5    W. Benson        21.4    +0.8      +9.8     ↑      │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌─── Risers (Biggest Score Increases) ────────────────────┐  │
│  │  Player         Previous   Current   Change              │  │
│  │  B. Baty        14.2       24.1      +9.9  🔥           │  │
│  │  C. Carroll     12.8       19.5      +6.7               │  │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  [View All Rankings]  [Export Data]  [Configure Alerts]       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack Recommendations

### Backend

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| Language | Python 3.11+ | pybaseball compatibility, data science ecosystem |
| Framework | FastAPI | Async support, auto-documentation, type hints |
| Database | PostgreSQL 15 | JSONB support, window functions, production-ready |
| Cache | Redis | Session storage, rate limiting, job queues |
| Job Queue | Celery + Redis | Scheduled calculations, alert processing |

### Frontend

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| Framework | React 18 | Component reuse, ecosystem maturity |
| Charts | Chart.js or Plotly | Already proven in Phase 1 |
| Styling | Tailwind CSS | Utility-first, matches glassmorphic design |
| State | React Query | Server state management, caching |

### Infrastructure

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| Hosting | AWS / GCP / Railway | Scalable, managed services |
| CI/CD | GitHub Actions | Integrated with repository |
| Monitoring | DataDog / New Relic | APM, alerting, logs |
| Storage | S3 / Cloud Storage | Historical data archives |

---

## Implementation Timeline (Phase 2)

| Week | Deliverables |
|------|--------------|
| 1-2 | Database migration (SQLite → PostgreSQL), API scaffolding |
| 3-4 | Data ingestion pipeline, rolling calculations |
| 5-6 | REST API endpoints, authentication |
| 7-8 | Dashboard UI, Chart integration |
| 9-10 | Alert system, email/webhook integration |
| 11-12 | Testing, deployment, documentation |

---

## Cost Estimates

### Monthly Operating Costs

| Service | Specification | Est. Cost |
|---------|--------------|-----------|
| Database | PostgreSQL (managed) | $25-50 |
| Compute | 2 vCPU, 4GB RAM | $20-40 |
| Redis | 1GB cache | $10-15 |
| Storage | 10GB | $1-2 |
| Monitoring | Basic tier | $0-25 |
| **Total** | | **$56-132/mo** |

---

## Security Considerations

1. **API Authentication:** JWT tokens with refresh rotation
2. **Rate Limiting:** 100 requests/minute per user
3. **Data Access:** Row-level security for user data
4. **Secrets Management:** Environment variables, no hardcoded keys
5. **HTTPS:** Required for all endpoints

---

## Open Questions for Phase 2

1. Should we support multiple weight configurations per user?
2. What's the SLA for alert delivery (real-time vs batch)?
3. Should historical data be purged after X years?
4. Integration with external tools (Slack, Discord)?

---

*Document prepared as part of MADDUX Phase 1 deliverables.*

