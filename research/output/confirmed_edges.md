# Confirmed Edges (out-of-sample + FDR survivors)


Grades: PLATINUM = passed all 5 gauntlet tests (purged CV, permutation, bootstrap, deflated Sharpe, PBO), net of transaction costs.


## gap_and_go

### Single factors


### Factor combinations


### Cumulative model (all factors)

- OOS rank IC: 0.019
- Top-decile OOS mean: 0.029%
- Top drivers: rvol_20d, nifty_ret_5d, atr14_pct, ret_20d, dist_prev_high, ret_5d, vix_level, rv20

### Hidden-edge detectors (level 4)

- [decay] ret_20d > 7.602: DECAYING (early 0.038% -> recent 0.008%)
- [decay] range_compression > 1.118: STABLE (early 0.020% -> recent 0.013%)
- [decay] up_streak > 1: DECAYING (early 0.029% -> recent 0.006%)
- [decay] ret_1d <= -1.688: STABLE (early 0.012% -> recent 0.013%)
- [decay] rvol_20d > 0.9274: STABLE (early 0.037% -> recent 0.033%)
- [decay] rvol_open30 > 1.322: STABLE (early 0.040% -> recent 0.036%)
- [decay] nifty_ret_5d <= -0.2228: STABLE (early 0.019% -> recent 0.011%)
- [decay] vix_level <= 15.4: DECAYING (early 0.014% -> recent 0.004%)

### Evolved alphas (level 6, GP-mined)

- `sub(rvol_open30, rank(sub(nifty_ret_5d, zscore_20d)))`: train IC 0.082, OOS IC 0.103 (n=3643)
- `rvol_20d`: train IC 0.050, OOS IC 0.093 (n=3643)
- `sub(rvol_20d, rank(sub(nifty_ret_5d, rvol_20d)))`: train IC 0.083, OOS IC 0.055 (n=3643)
- `sub(rvol_20d, rank(nifty_ret_5d))`: train IC 0.081, OOS IC 0.051 (n=3643)
- `sub(rvol_20d, rank(sub(nifty_ret_5d, mkt_gap_median)))`: train IC 0.075, OOS IC 0.049 (n=3643)
- `sub(sub(nifty_ret_5d, rvol_20d), rvol_open30)`: train IC 0.080, OOS IC -0.048 (n=3643)
- `sub(rvol_20d, rank(sub(nifty_ret_5d, zscore_20d)))`: train IC 0.096, OOS IC 0.045 (n=3643)
- `sub(rvol_20d, rank(sub(nifty_ret_5d, gap_idiosyncratic)))`: train IC 0.067, OOS IC 0.045 (n=3643)
- `sub(nifty_ret_5d, rvol_open30)`: train IC 0.078, OOS IC -0.043 (n=3643)
- `sub(rvol_20d, rank(sub(nifty_ret_5d, nifty_ret_1d)))`: train IC 0.077, OOS IC 0.041 (n=3643)

### Validation gauntlet (level 5) -- FINAL GRADES

- **SILVER** L6:sub(rvol_open30, rank(sub(nifty_ret_5d, zscore_20d))): net mean -0.004%/trade, folds 5/5, perm p=0.002, PBO=0.429, DSR=0.0003 [3/5]
- **SILVER** L6:rvol_20d: net mean -0.024%/trade, folds 4/5, perm p=0.014, PBO=0.429, DSR=0.0 [3/5]
- **SILVER** L6:sub(rvol_20d, rank(sub(nifty_ret_5d, rvol_20d))): net mean 0.003%/trade, folds 4/5, perm p=0.002, PBO=0.429, DSR=0.0014 [3/5]
- **SILVER** L6:sub(rvol_20d, rank(nifty_ret_5d)): net mean 0.008%/trade, folds 5/5, perm p=0.002, PBO=0.429, DSR=0.0043 [3/5]
- **SILVER** L6:sub(rvol_20d, rank(sub(nifty_ret_5d, mkt_gap_median))): net mean 0.002%/trade, folds 5/5, perm p=0.002, PBO=0.429, DSR=0.0011 [3/5]
- **SILVER** L6:sub(sub(nifty_ret_5d, rvol_20d), rvol_open30): net mean -0.012%/trade, folds 5/5, perm p=0.004, PBO=0.429, DSR=0.0 [3/5]
- **SILVER** L6:sub(rvol_20d, rank(sub(nifty_ret_5d, zscore_20d))): net mean -0.011%/trade, folds 5/5, perm p=0.004, PBO=0.429, DSR=0.0001 [3/5]
- **SILVER** L6:sub(rvol_20d, rank(sub(nifty_ret_5d, gap_idiosyncratic))): net mean 0.010%/trade, folds 5/5, perm p=0.002, PBO=0.429, DSR=0.0062 [3/5]
- **SILVER** L6:sub(nifty_ret_5d, rvol_open30): net mean -0.012%/trade, folds 5/5, perm p=0.004, PBO=0.429, DSR=0.0 [3/5]
- **SILVER** L6:sub(rvol_20d, rank(sub(nifty_ret_5d, nifty_ret_1d))): net mean -0.013%/trade, folds 4/5, perm p=0.004, PBO=0.429, DSR=0.0 [3/5]

## mean_reversion

### Single factors


### Factor combinations


### Cumulative model (all factors)

- OOS rank IC: 0.014
- Top-decile OOS mean: 0.007%
- Top drivers: rvol_open30, dist_prev_low, rvol_20d, dist_prev_high, ret_20d, vix_level, gap_pct, zscore_20d

### Hidden-edge detectors (level 4)

- [decay] rvol_open30 <= 0.392: STABLE (early 0.018% -> recent 0.021%)
- [decay] nifty_ret_1d > 1.205: STABLE (early 0.023% -> recent 0.025%)
- [decay] gap_abs <= 0.8338: STABLE (early 0.011% -> recent 0.010%)
- [decay] gap_vs_atr <= 0.7264: STRENGTHENING (early 0.004% -> recent 0.006%)
- [decay] gap_idiosyncratic > -1.285: STABLE (early 0.005% -> recent 0.006%)
- [decay] ret_1d <= 1.926: STABLE (early 0.004% -> recent 0.005%)
- [decay] dist_prev_low <= 2.39: STABLE (early 0.004% -> recent 0.006%)
- [decay] vix_level <= 19.63: STABLE (early 0.005% -> recent 0.005%)