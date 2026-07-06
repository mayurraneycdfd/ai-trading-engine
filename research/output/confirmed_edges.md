# Confirmed Edges (out-of-sample + FDR survivors)


Grades: PLATINUM = passed all 5 gauntlet tests (purged CV, permutation, bootstrap, deflated Sharpe, PBO), net of transaction costs.


## gap_and_go

### Single factors


### Cumulative model (all factors)

- OOS rank IC: 0.024
- Top-decile OOS mean: 0.016%
- Top drivers: rvol_open30, nifty_ret_5d, zscore_20d, dist_prev_high, rv20, ret_20d, vix_level, atr14_pct

### Hidden-edge detectors (level 4)

- [decay] rvol_open30 > 1.322: STABLE (early 0.040% -> recent 0.036%)

### Evolved alphas (level 6, GP-mined)

- `rvol_open30`: train IC 0.039, OOS IC 0.097 (n=3643)
- `ts_mean5(rvol_open30)`: train IC 0.029, OOS IC 0.054 (n=3611)

### Validation gauntlet (level 5) -- FINAL GRADES

- **REJECTED** L6:rvol_open30: net mean -0.955%/trade, folds 0/5, perm p=0.02, PBO=0.0, DSR=0.0 [2/5]
- **REJECTED** L6:ts_mean5(rvol_open30): net mean -0.963%/trade, folds 0/5, perm p=0.1018, PBO=0.0, DSR=0.0 [1/5]

## mean_reversion

### Single factors


### Factor combinations


### Cumulative model (all factors)

- OOS rank IC: 0.009
- Top-decile OOS mean: 0.006%
- Top drivers: dist_prev_low, ret_20d, dist_prev_high, nifty_ret_1d, day_of_week, rvol_open30, up_streak, ret_5d

### Hidden-edge detectors (level 4)
