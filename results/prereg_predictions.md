# Pre-Registered Predictions — Out-of-Sample CR–Seismic Validation

**Written:** 2026-04-24T12:05:55Z
**Git SHA:** 817d7ba
**OOS window:** 2020-01-01 → 2025-04-29
**Surrogates:** 100,000 phase-randomisation

This file was created BEFORE loading or analysing any out-of-sample data.
All thresholds are pre-specified.  Results are recorded in
`results/out_of_sample_report.md`.

---

## In-sample context (1976–2019)

From scripts 02–05 (Homola replication + stress tests):

| Quantity | Value |
|---|---|
| Dominant peak lag (raw) | −525 days (half solar cycle) |
| Dominant peak \|r\| (raw) | 0.469 |
| r(τ=+15 d) raw | +0.310 (solar-cycle confounded) |
| r(τ=+15 d) HP-detrended | +0.041 |
| In-sample p_global (IAAFT, raw) | 1.000 (NOT significant after surrogate correction) |
| After detrending | p < 0.001 at lags ≠ +15 d |

The in-sample dominant peak is at −525 days, not at the claimed +15 days.
r(+15 d) ≈ 0.04 after solar-cycle removal — this is the baseline expectation
for the out-of-sample window.

---

## Pre-registered predictions

### P1 — Sign and location of claimed correlation peak
**Prediction:** If Homola et al.'s mechanism is real, the OOS window should show
a cross-correlation peak at τ ≈ +15 days (cosmic rays leading seismic activity
by 15 days) with **positive sign** (positive CR deviation → elevated seismic
Mw-sum 15 days later).

**Operationalisation:**
- PASS if r(τ=+15 d) > 0 AND the lag of maximum |r(τ)| for τ ∈ [5, 30] days
  is within ±3 days of +15 days.
- FAIL otherwise.

**Baseline from in-sample HP-detrended:** r(+15 d) ≈ +0.041
**Monte Carlo tolerance (at 100,000 surrogates):** ±0.0063

### P2 — Significance and solar-phase trend
**Prediction:** The OOS window (2020–2025) covers Solar Cycle 25
rising phase, approaching the predicted 2025–2027 solar maximum.  Homola's
model predicts the CR–seismic correlation should be in a RISING phase of its
~11-year envelope (the last in-sample envelope peak was near 2014).

**Operationalisation:**
- PASS if: (a) p_global (phase-surrogate) < 0.05, AND
  (b) r(τ=+15 d) in rolling 18-month windows shows a non-negative trend
  (slope ≥ 0) across the OOS period.
- PARTIAL if (a) holds but (b) does not.
- FAIL if p_global ≥ 0.05.

### P3 — Rolling-window lag stability
**Prediction:** The lag at which r(τ) is maximised for τ ∈ [5, 30] days should
be stable to within ±3 days across 18-month rolling windows of the OOS data.

**Operationalisation:**
- PASS if std(τ*) ≤ 5 days across rolling sub-windows where a peak
  in [5, 30] days exists.
- FAIL if std(τ*) > 10 days or peaks migrate outside [5, 30] days in majority
  of windows.

### P4 — Geographic non-localisation
**Prediction:** Per Homola et al.'s own result, the correlation should be GLOBAL
(disappear in location-specific analyses).  After BH FDR correction at q=0.05,
the number of significant (station, cell) pairs should NOT significantly exceed
the expected false-discovery count.

**Operationalisation:**
- PASS if n_significant ≤ 2 × expected_FP (BH q=0.05).
- FAIL if n_significant > 2 × expected_FP AND a clear geographic cluster emerges.

---

## Falsification criteria (pre-specified)

### F1 — No peak in claimed window
**Criterion:** No lag τ ∈ [5, 30] days has |r(τ)| exceeding the 95th percentile
of the phase-surrogate distribution.

- F1 TRIGGERED (Homola falsified) if the criterion holds across the full OOS
  window AND across all 18-month sub-windows.

### F2 — Peak lag drift
**Criterion:** The optimal lag τ* for τ ∈ [5, 30] days drifts by more than
±10 days between any two adjacent 18-month rolling windows.

- F2 TRIGGERED if drift > 10 days in majority of window pairs.

### F3 — Unexpected geographic localisation
**Criterion:** The OOS correlation is STRONGER in a specific geographic region
than globally — the inverse of Homola's own finding.

- F3 TRIGGERED if n_significant > 3 × expected_FP AND a geographic cluster
  with min p < BH-threshold is identified.
- This would be informative negative evidence: a real local effect, but NOT
  the global cosmic-ray mechanism Homola proposed.

---

## Analysis decisions (pre-specified)

| Parameter | Value | Reason |
|---|---|---|
| Bin size | 5 days | Matches Homola et al. |
| Lag range | ±200 days | Covers claimed +15 d with context; shorter window makes ±1000 d infeasible |
| Surrogates | 100,000 | GPU-accelerated; MC tolerance ±0.0063 |
| Surrogate method | Phase randomisation | Preserves power spectrum; faster than IAAFT |
| Detrending | Linear + sunspot OLS | HP/STL inappropriate for <1 solar cycle window |
| Min stations/bin | 3 | Matches Homola et al. |
| Min magnitude | 4.0 | Matches Homola et al. |
| Rolling window | 18 months | Minimum for meaningful correlation at 5-day bins |
| Rolling step | 3 months | Smooth time evolution |
| FDR | BH q=0.05 | Standard |

---
*This file is part of a pre-registered analysis. Results are reported regardless
of direction in `results/out_of_sample_report.md`.*
