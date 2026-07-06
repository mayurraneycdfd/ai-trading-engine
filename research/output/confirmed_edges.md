# Confirmed Edges (out-of-sample + FDR survivors)


Grades: PLATINUM = passed all 5 gauntlet tests (purged CV, permutation, bootstrap, deflated Sharpe, PBO), net of transaction costs.


## gap_and_go

### Single factors


### Cumulative model (all factors)

- OOS rank IC: 0.027
- Top-decile OOS mean: 0.005%
- Top drivers: rvol_open30, dist_prev_low, ms_amihud_z20, ms_roll_spread_z20, zscore_20d, ms_vpin_z20, ms_clv_pressure_z20, ret_5d

### Hidden-edge detectors (level 4)

- [decay] rvol_open30 > 1.322: STABLE (early 0.040% -> recent 0.036%)

## mean_reversion

### Single factors


### Factor combinations


### Cumulative model (all factors)

- OOS rank IC: 0.005
- Top-decile OOS mean: 0.022%
- Top drivers: ms_kyle_lambda_z20, ms_kyle_lambda, dist_prev_low, ms_amihud, mkt_gap_median, nifty_ret_1d, ms_amihud_z20, vix_level

### Hidden-edge detectors (level 4)
