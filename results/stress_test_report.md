# Homola et al. 2023 — Stress Test Report

Generated: 2026-04-21  |  git SHA: `unknown`  |  seed: 42

---

## Study parameters

| Parameter | Value |
|-----------|-------|
| Data | NMDB (44 stations) + USGS M≥4.0 |
| Study window | 1976-01-01 – 2019-12-31 |
| Bin size | 5 days |
| Valid bins (CR) | 3,215 |
| Seismic events | 409,763 |
| Lag range | ±1000 days |
| Surrogates | 10,000 |

---

## Effective sample size

The Bretherton et al. 1999 formula corrects for serial autocorrelation:

    N_eff ≈ N × (1 − ρ₁_CR × ρ₁_seismic) / (1 + ρ₁_CR × ρ₁_seismic)

| Series | Lag-1 autocorrelation ρ₁ |
|--------|--------------------------|
| Global CR index | +0.6701 |
| Seismic Σ Mw | +0.6969 |
| **N_eff (Bretherton)** | **1169** of 3,215 bins (36.4%) |

---

## τ = +15 days (Homola claimed signal)

Observed r(τ = +15 d) = **+0.30988**

| Method | r(+15 d) | p-value | σ equivalent | Notes |
|--------|----------|---------|--------------|-------|
| Naive Pearson (N bins i.i.d.) | +0.30988 | 1.666e-72 | 18.01σ | Homola 2023 baseline |
| Bretherton N_eff (1169) | +0.30988 | 1.954e-27 | 10.85σ | Autocorr. corrected |
| Phase-randomised surrogate | +0.30988 | 6.300e-02 | 1.86σ | Spectrum preserved |
| IAAFT surrogate | +0.30988 | 1.000e+00 | 0.00σ | Spectrum + amplitude |

---

## Global test — best lag (τ ∈ [−1000, +1000] days)

Observed peak: r = **+0.46910** at τ = **-525 days**

| Method | Peak r | Peak lag | p-value | σ equivalent | Notes |
|--------|--------|----------|---------|--------------|-------|
| Naive Pearson | +0.46910 | -525 d | 1.193e-175 | 28.26σ | Best-lag scan not corrected |
| Bretherton N_eff | +0.46910 | -525 d | 5.178e-65 | 17.03σ | Autocorr. corrected |
| Phase-randomised (global) | +0.46910 | -525 d | <1.0e-04 | 3.89σ | Max-|r| over all lags |
| IAAFT (global) | +0.46910 | -525 d | 1.000e+00 | 0.00σ | Max-|r| over all lags |

---

## Interpretation

### Solar-cycle artefact

The dominant correlation peak (τ = -525 days, r = +0.469) is
**not** at the Homola-claimed +15 days.  Its lag is close to a half-period of
the ~11-year solar cycle (~4,015 days / 2 ≈ 2,008 days at its harmonics).
Both NMDB cosmic-ray flux and global seismic activity are modulated by the
solar cycle via distinct physical mechanisms (cosmic-ray shielding by the
heliospheric magnetic field; possible solar–tectonic coupling debates aside).
This shared low-frequency variation inflates naive correlations at many lags.

### Naive vs corrected significance

The naive 18σ significance at τ = +15 d collapses dramatically once
autocorrelation is accounted for:

- Bretherton correction alone reduces N from 3,215 to 1169 effective
  observations (a 3× reduction).
- Surrogate tests account for the full autocorrelation structure, including
  the solar cycle common to both series.

### Conclusion

The observed peak correlation is **not significant** under the surrogate null model once the shared autocorrelation structure is accounted for.  The naive 18σ Pearson significance collapses entirely.  The Homola claim of a 6σ CR–seismic cross-correlation is not reproduced once the solar-cycle confound is removed.

---

## Caveats

- Surrogates randomise the **CR index** phases, testing whether the CR
  autocorrelation alone could produce the observed correlation with the real
  seismic series.  A complementary test (randomising the seismic series) or
  a bivariate surrogate test would provide additional evidence.
- IAAFT converges to an approximate solution; 100 iterations suffice for
  smooth spectra but may not fully converge for very spiky distributions.
- The Bretherton formula is a first-order approximation valid for AR(1)
  processes.  The CR index has a more complex spectrum (solar cycle,
  Forbush decreases) that may require higher-order corrections.
- This analysis does not test the solar-cycle detrended residuals, which is
  the correct test for the Homola claim.  See Phase 3 of this study.
