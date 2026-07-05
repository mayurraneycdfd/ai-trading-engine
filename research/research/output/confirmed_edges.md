# Confirmed Edges (out-of-sample + FDR survivors)


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