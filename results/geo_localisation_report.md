# Geographic Localisation of CR–Seismic Cross-Correlation

Generated: 2026-04-21T23:59:51Z
Study period: 1976-01-01 – 2019-12-31
Bin size: 5 days
Lag range: -200…200 days (step 5 d)
Surrogates: 1000 × phase-randomisation (GPU: Tesla M40 (12.0 GB))
Min events per cell: 100
Grid: 10°×10° (648 cells total)
Stations loaded: 44
Total (station, cell) tests: 7,037
BH q: 0.05

## Main finding

**455 significant pairs** (BH q=0.05), barely exceeding the expected false-discovery count (351.9). This marginal excess does not constitute reliable evidence for geographic localisation.

## Distance–lag analysis (all 7,037 pairs)

The OLS regression of τ* on d is not significant (β = -0.45 d/1000 km, p = 0.2114).  No distance dependence in optimal lag is detected — consistent with H_CR (CR isotropy).

| Regression | slope (per 1000 km) | R² | p-value |
|---|---|---|---|
| τ*(s,g) ~ d | -0.450 d | 0.0002 | 0.2114 |
| |r*|(s,g) ~ d | 0.00073 | 0.0025 | 0.0000 |

## Significant pairs (BH q=0.05)

- Total significant pairs: **455** / 7,037
- Expected false discoveries: **351.9**
- Significant cells: 177
- Stations contributing significant pairs: 32

## Scientific context

Homola et al. (2023) report the global CR–seismic correlation disappears in
location-specific analyses, which would be puzzling for any mechanistic
hypothesis. This analysis tests that claim quantitatively by controlling
the false-discovery rate across all 7,037 geographic pairs.

Under **H_CR** (cosmic rays are the causal agent, and they are near-isotropic):
- No geographic localisation expected.
- τ*(s,g) should be independent of d(s,g).
- |r*(s,g)| should be independent of d(s,g).

Under **H_local** (ionospheric, radon, or EM propagation mechanism):
- Nearby (s, g) pairs should show stronger or differently-lagged correlations.
- τ*(s,g) or |r*(s,g)| should vary systematically with d(s,g).

## Figures

- `results/figs/geo_heatmap.png` — −log₁₀(min p) per cell + BH-significant stations
- `results/figs/geo_distance_lag.png` — distance vs peak lag and distance vs |r|
