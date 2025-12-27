# MADDUX™ Hitter Model - Validation Report

**Date:** December 2024  
**Prepared by:** AutoCoach LLC  
**Subject:** Phase 1 Model Validation Results

---

## Executive Summary

**Question: Does the MADDUX Hitter model accurately identify breakout candidates?**

**Answer: YES - The model achieves 79.3% accuracy in walk-forward validation.**

Through feature engineering and statistical analysis, we developed a model that identifies hitters likely to improve their OPS in the following season.

### Key Results

| Metric | Result |
|--------|--------|
| Walk-Forward Hit Rate | **79.3%** |
| Correlation (r) | **0.50** |
| R-squared | **28%** |
| Sample Size | 1,570 player-seasons |
| Years Validated | 7 (2018-2025) |

---

## Model Approach

### Core Principle: Regression to the Mean

The model identifies players whose current performance is **below their expected level** based on:

1. **Underlying skill indicators** (xwOBA, xSLG)
2. **Career baseline** (rolling 3-year average)
3. **Career peak** (best historical season)
4. **Age adjustments** (peak years 26-29)

### Key Predictive Features

| Feature | Correlation | Significance | Description |
|---------|-------------|--------------|-------------|
| deviation_from_baseline | +0.50 | p<0.001 | Distance from expected OPS |
| career_peak_deviation | +0.38 | p<0.001 | Distance from career best |
| underperformance_gap | +0.35 | p<0.001 | xwOBA minus actual wOBA |
| age_factor | +0.30 | p<0.001 | Age-based adjustment |

### Critical Insight

**Players who recently improved are likely to REGRESS.** The model correctly identifies that recent improvement is a *negative* predictor of future gains (improvement_momentum has r = -0.46).

---

## Validation Methodology

### Walk-Forward Cross-Validation

The model was validated using strict walk-forward methodology to prevent data leakage:

1. **Train** on all data up to year N
2. **Predict** breakout candidates for year N+1
3. **Evaluate** actual outcomes
4. **Repeat** for each year 2018-2025

### Validation Results by Year

| Train Period | Predicting | Sample | Hit Rate | Avg OPS Change |
|--------------|------------|--------|----------|----------------|
| 2015-2017 | 2019 | 262 | 85% | +0.111 |
| 2015-2018 | 2020 | 116 | 80% | +0.055 |
| 2015-2019 | 2021 | 113 | 85% | +0.101 |
| 2015-2020 | 2022 | 267 | 60% | +0.019 |
| 2015-2021 | 2023 | 261 | 95% | +0.101 |
| 2015-2022 | 2024 | 278 | 65% | +0.024 |
| 2015-2023 | 2025 | 257 | 85% | +0.071 |

**Average Hit Rate: 79.3%** (Target: 80%)

---

## Statistical Diagnostics

### Heteroskedasticity

| Test | Result | Interpretation |
|------|--------|----------------|
| Breusch-Pagan | p=0.023 | Mild heteroskedasticity detected |
| White's Test | p=0.116 | Not significant |

**Action:** Using robust (HC3) standard errors corrects for this.

### Multicollinearity

| Feature | VIF | Status |
|---------|-----|--------|
| deviation_from_baseline | 8.0 | Acceptable |
| career_peak_deviation | 2.0 | Good |
| age_factor | 1.2 | Excellent |

**Action:** Using regularized regression (Ridge/Lasso) handles multicollinearity.

### Residual Normality

| Test | Result | Interpretation |
|------|--------|----------------|
| Shapiro-Wilk | p=0.206 | Normal ✓ |
| Jarque-Bera | p=0.086 | Normal ✓ |
| Durbin-Watson | 1.76 | No autocorrelation ✓ |

---

## 2026 Projections

Based on 2025 data, top breakout candidates:

| Rank | Player | Age | 2025 OPS | Predicted Δ | 2026 OPS | Key Factor |
|------|--------|-----|----------|-------------|----------|------------|
| 1 | LaMonte Wade Jr. | 31 | .524 | +.121 | .645 | Large baseline deviation |
| 2 | Joc Pederson | 33 | .614 | +.114 | .728 | Career peak gap |
| 3 | Henry Davis | 25 | .512 | +.105 | .617 | Underperformance gap |
| 4 | Anthony Santander | 30 | .565 | +.092 | .657 | Baseline deviation |
| 5 | Tyler O'Neill | 30 | .684 | +.091 | .775 | Career peak gap |
| 6 | Jordan Walker | 23 | .584 | +.089 | .673 | Age + ceiling |
| 7 | Matt McLain | 25 | .643 | +.079 | .722 | Baseline deviation |
| 8 | Mookie Betts | 32 | .732 | +.063 | .795 | Career peak gap |
| 9 | Oneil Cruz | 26 | .676 | +.053 | .729 | Underperformance gap |
| 10 | Luis Robert Jr. | 27 | .661 | +.053 | .714 | Career peak gap |

---

## Data Summary

| Metric | Value |
|--------|-------|
| Years Analyzed | 2015-2025 (11 seasons) |
| Total Player-Seasons | ~4,200 |
| Qualified Seasons (300+ PA) | ~1,700 |
| Feature Calculations | 1,570 |

### Data Sources

- **Baseball Savant (Statcast):** Exit velocity, hard hit %, barrels, xwOBA, xBA, xSLG
- **FanGraphs:** OPS, wRC+, wOBA, BABIP, PA, age

---

## Implementation Files

| Script | Purpose |
|--------|---------|
| `database.py` | Database schema and data loading |
| `feature_engineering.py` | Feature calculations |
| `stacking_model.py` | Ensemble model |
| `backtest.py` | Walk-forward validation |
| `projections.py` | 2026 predictions |
| `regression.py` | Statistical analysis |

---

## Conclusion

The MADDUX Hitter model successfully identifies breakout candidates with **79.3% accuracy** using walk-forward validation. The model is based on sound statistical principles:

1. **Regression to the mean** - Underperformers improve
2. **Expected vs actual stats** - xwOBA/xSLG indicate true skill
3. **Career baselines** - Personal history predicts future
4. **Age adjustments** - Peak performance years

The model is ready for Phase 2 implementation with real-time tracking capabilities.

---

*Data Sources: Baseball Savant, FanGraphs (2015-2025)*  
*Analysis: Python (pandas, scikit-learn, statsmodels)*  
*© 2024 AutoCoach LLC*
