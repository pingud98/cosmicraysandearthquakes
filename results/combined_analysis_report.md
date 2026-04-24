# Combined Full-Series Analysis (1976–2025)

Generated: 2026-04-24T05:33:39Z
Full window: 1976-01-01 → 2025-04-29
In-sample: 1976-01-01 → 2019-12-31
Out-of-sample: 2020-01-01 → 2025-04-29
GPU: Tesla M40 (12.0 GB)
Surrogates: 10,000 per window

## Does appending OOS data strengthen or weaken significance?

| Window | p_global | σ_surrogate | peak lag |
|---|---|---|---|
| In-sample (1976–2019) | 0.0394 | 2.06 | -125 d |
| Out-of-sample (2020–2025) | N/A | N/A | None d |
| Combined (1976–2025) | 0.0391 | 2.06 | -125 d |

## Sinusoidal envelope fit

BF = 27.45: strong evidence for sinusoidal envelope

Best-fit period: **9.95 years** (constrained to [9, 13] years)

| Parameter | Value |
|---|---|
| Period P | 9.95 yr |
| Amplitude A | 0.1470 |
| Phase φ | 4.41 rad |
| Baseline μ | 0.0481 |
| Model B BIC | -153.76 |
| Model A BIC | -147.14 |
| ΔBIC (A−B) | 6.62 |
| Bayes factor (BF) | 27.451 |

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
