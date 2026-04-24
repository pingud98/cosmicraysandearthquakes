# Out-of-Sample Validation Report — Homola et al. 2023

Generated: 2026-04-24T12:06:46Z
Git SHA: 817d7ba
OOS window: 2020-01-01 → 2025-04-29
Analysis run date: 2026-04-24
Data availability check: 2026-04-22

## Overall verdict

**AMBIGUOUS**: Mixed results; insufficient evidence to confirm or refute.

## Prediction scorecard

| Criterion | Outcome |
|---|---|
| P1 | PASS |
| P2 | FAIL |
| P3 | AMBIGUOUS |
| P4 | AMBIGUOUS |
| F1 | TRIGGERED |
| F2 | not triggered |
| F3 | AMBIGUOUS |

## Key numerical results

| Metric | OOS value | In-sample baseline |
|---|---|---|
| r(τ = +15 d) raw | +0.0304 | +0.3099 (solar-cycle confounded) |
| r(τ = +15 d) HP-detrended | +0.0232 | +0.0411 |
| Surrogate 95th pct at τ=+15 d | 0.1012 | (not computed in-sample at this lag) |
| p_global (phase surrogates) | 0.1002 | 1.000 (in-sample raw, not significant) |
| σ_surrogate | 1.64 | n/a |
| Dominant peak lag | +125 d | −525 d |
| Dominant peak \|r\| | 0.1358 | 0.469 |
| BH-significant pairs (geo) | 0 | 455 (in-sample) |
| Expected FP (geo, BH q=0.05) | 0.0 | 351.9 (in-sample) |
| Surrogate count | 100,000 | 10,000 (in-sample) |

## Interpretation notes

The OOS window (2020-01-01–2025-04-29) spans approximately
5 years —
less than one full 11-year solar cycle.  This has two implications:

1. **Solar-cycle detrending is less effective** over sub-cycle windows.  Linear
   and sunspot-regression detrending are used instead of HP/STL, which require
   series longer than the target period.

2. **Statistical power is lower** than in-sample (T ≈ 3215 bins vs
   T ≈ 390 bins OOS).  A genuine effect of the same magnitude as the
   in-sample HP-detrended signal (r ≈ 0.04) would require a very large n_surr
   to detect reliably.

## Methodological notes

- Pre-registration file: `results/prereg_predictions.md` (timestamps confirm
  it was written before any OOS analysis was run)
- GPU: CuPy not installed
- Surrogates: phase-randomisation (100,000)
- Lag range: ±200 days

## Figures

- `results/figs/oos_xcorr.png` — r(τ) with surrogate envelopes
- `results/figs/rolling_correlation_oos.png` — rolling r(τ=+15 d)
