# Detrended CR–Seismic Cross-Correlation Analysis

Generated: 2026-04-24T12:15:00Z
Study period: 1976-01-01 – 2019-12-31
Bin size: 5 days
Surrogates: 10000 IAAFT
Lag range: -1000…1000 days

## Significance table

| Method | N_eff | r(+15d) | σ_Breth(15d) | Peak r | Peak lag | p_global (IAAFT) | σ_IAAFT |
|---|---|---|---|---|---|---|---|
| Raw | 2916 | 0.0815 | 4.41 | 0.1386 | -525 d | 0.0000 | 3.9σ |
| HP filter | 3199 | 0.0266 | 1.5 | 0.1009 | -125 d | 0.0000 | 3.9σ |
| STL | 3031 | 0.0296 | 1.63 | 0.0934 | -525 d | 0.0000 | 3.9σ |
| Sunspot regression | 3056 | 0.0368 | 2.03 | 0.0919 | -125 d | 0.0000 | 3.9σ |

## Interpretation

**CAUTION**: The following variants retain p_global < 0.05 after detrending: Raw, HP filter, STL, Sunspot regression.  Further investigation required.

## Methods

### HP filter (Hodrick-Prescott)
λ = 1.29e+05 calibrated for 5-day bins targeting removal of
variations longer than ~2000 days (Ravn & Uhlig 2002 scaling of the standard λ=1600).

### STL decomposition
Period = 803 bins ≈ 11.0 years
(11-year solar cycle). seasonal_jump=100, trend_jump=100 for computational efficiency.
The *residual* component (x − trend − seasonal) is used.

### Sunspot OLS regression
Contemporaneous + [0, 30, 90, 180]-day lagged sunspot numbers from SIDC WDC-SILSO.
The fitted solar-proxy component is subtracted from each series.

## Figure
`results/figs/detrended_xcorr.png`
