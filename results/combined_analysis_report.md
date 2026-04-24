# Combined Full-Series Analysis (1976–2025)

Generated: 2026-04-24T12:08:35Z
Full window: 1976-01-01 → 2025-04-29
In-sample: 1976-01-01 → 2019-12-31
Out-of-sample: 2020-01-01 → 2025-04-29
GPU: CuPy not installed
Surrogates: 10,000 per window

## Does appending OOS data strengthen or weaken significance?

| Window | p_global | σ_surrogate | peak lag |
|---|---|---|---|
| In-sample (1976–2019) | 0.0010 | 3.29 | -125 d |
| Out-of-sample (2020–2025) | 0.1053 | 1.62 | 125 d |
| Combined (1976–2025) | 0.0102 | 2.57 | -125 d |

## Sinusoidal envelope fit

BF = 0.75 < 1: evidence FAVOURS constant model (no envelope)

Best-fit period: **13.00 years** (constrained to [9, 13] years)

| Parameter | Value |
|---|---|
| Period P | 13.00 yr |
| Amplitude A | 0.0473 |
| Phase φ | 3.43 rad |
| Baseline μ | 0.0211 |
| Model B BIC | -240.09 |
| Model A BIC | -240.66 |
| ΔBIC (A−B) | -0.57 |
| Bayes factor (BF) | 0.752 |

## Station roster comparison (OOS window)

| Roster | Description | Stations | p_global |
|---|---|---|---|
| A | In BOTH windows | ? | N/A |
| B_oos | All OOS stations | ? | N/A |
| C | New OOS-only | ? | N/A |

A real effect should appear consistently across all three rosters.
Divergence (e.g., significant only in A) would suggest station-selection bias.

## Figure
`results/figs/full_series_with_envelope_fit.png`
