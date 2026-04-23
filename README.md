# Cosmic Rays and Earthquakes: A Rigorous Replication Study

A reproducible, GPU-accelerated statistical pipeline that tests the claimed
correlation between galactic cosmic-ray flux and global seismicity
([Homola et al. 2023](https://doi.org/10.3390/rs15010200)).

---

## Summary of findings

| Stage | Key result |
|---|---|
| In-sample replication (1976–2019) | r(+15 d) = +0.31 **raw**; drops to **+0.04** after solar-cycle detrending |
| Global surrogate test (IAAFT, 100 k surrogates) | p = 1.00 after detrending — **not significant** |
| Geographic localisation (34 stations × 207 cells) | No distance–lag dependence; β = −0.45 d/1000 km, p = 0.21 |
| Out-of-sample validation (2020–2025) | Results in `results/out_of_sample_report.md` |

The raw r = 0.31 is an artefact of the shared ~11-year solar cycle modulating
both cosmic-ray flux and seismicity.  After removing this trend the signal
is indistinguishable from phase-randomised noise.

---

## Repository structure

```
scripts/          Analysis pipeline (run in order)
  01_download_data.py            Download NMDB / USGS / SIDC data
  02_homola_replication.py       Replicate Homola et al. cross-correlation
  03_stress_test.py              Surrogate significance test (CPU + GPU)
  04_detrended_analysis.py       HP-filter / sunspot detrending
  05_geographic_localisation.py  Station × grid-cell BH-FDR scan
  06_check_data_availability.py  Determine reliable OOS data window
  07_out_of_sample.py            Pre-registered out-of-sample validation
  08_combined_timeseries.py      1976-to-present sinusoid fit + Bayes factor
  benchmark_gpu.py               GPU vs CPU surrogate benchmark

src/crq/          Python package
  ingest/         NMDB, USGS, SIDC, station-roster loaders
  preprocess/     Hodrick-Prescott and linear detrending
  stats/          Phase-randomisation / IAAFT surrogates (CPU + GPU)

results/          Generated outputs (committed)
  prereg_predictions.md          Pre-registration (timestamped before OOS run)
  data_availability.json         Reliable data window determination
  homola_replication.json        In-sample cross-correlation results
  detrended_results.json         Post-detrending results
  geo_localisation.json          Geographic localisation scan
  out_of_sample_metrics.json     OOS validation metrics (post-run)
  figs/                          Plots

config/
  stations.yaml   NMDB station list with coordinates
tests/            pytest suite (29 tests)
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/pingud98/cosmicraysandearthquakes.git
cd cosmicraysandearthquakes
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Download data (NMDB, USGS M≥4.5, SIDC sunspots)
python scripts/01_download_data.py

# 3. Run in-sample analysis
python scripts/02_homola_replication.py
python scripts/03_stress_test.py --n-surrogates 10000
python scripts/04_detrended_analysis.py

# 4. Geographic scan (GPU recommended)
python scripts/05_geographic_localisation.py --n-surrogates 1000

# 5. Check data availability for out-of-sample window
python scripts/06_check_data_availability.py

# 6. Pre-registered out-of-sample validation (writes prereg BEFORE analysis)
python scripts/07_out_of_sample.py --study-start 2020-01-01 --study-end 2025-04-29

# 7. Combined timeseries with sinusoid fit
python scripts/08_combined_timeseries.py
```

GPU (CUDA) is used automatically when available. Scripts fall back to CPU
with a warning. The surrogate tests were benchmarked on a Tesla M40 (12 GB):

| Method | CPU | GPU | Speedup |
|---|---|---|---|
| Phase randomisation | 61.7 s | 20.9 s | 2.9× |
| IAAFT | 227.8 s | 175.6 s | 1.3× |

---

## Data sources

| Source | Content | Access |
|---|---|---|
| [NMDB](https://www.nmdb.eu) | Hourly neutron monitor counts, pressure-corrected | Free, HTTP |
| [USGS FDSN](https://earthquake.usgs.gov/fdsnws/event/1/) | M ≥ 4.5 global catalogue | Free, HTTP |
| [SIDC SILSO](https://www.sidc.be/silso/datafiles) | Daily international sunspot number | Free, HTTP |

Data are downloaded by the scripts and cached locally in `data/`.
No data files are committed to this repository.

---

## Pre-registration

`results/prereg_predictions.md` was committed to git **before** any
out-of-sample data were loaded (UTC 2026-04-22T00:44:30, commit `1832f73`).
This prevents post-hoc hypothesis adjustment.  Verify with:

```bash
git log --diff-filter=A results/prereg_predictions.md
```

---

## Statistical methods

- **Surrogate test**: Phase randomisation preserves the power spectrum of the
  cosmic-ray series; 100,000 surrogates give p-value resolution of 10⁻⁵.
- **IAAFT**: Iterated amplitude-adjusted FT surrogates (preserves amplitude
  distribution as well as power spectrum).
- **Detrending**: Hodrick-Prescott filter (λ = 1.29 × 10⁵) for in-sample
  window; linear detrending for OOS (< 1 solar cycle).
- **FDR control**: Benjamini-Hochberg at q = 0.05 for the geographic scan.
- **Bayes factor**: BIC approximation comparing sinusoidal vs constant model
  on the full 1976-to-present correlation timeseries.

---

## Requirements

- Python ≥ 3.10
- numpy, pandas, scipy, matplotlib, pyyaml, requests
- CuPy ≥ 12 (optional, for GPU acceleration)

Install: `pip install -e .`

---

## Citation

If you use this pipeline please cite:

> Homola P. et al. (2023). *Indication of Correlation between Cosmic-Ray
> Flux and Lightning Activity*. Remote Sensing 15(1), 200.
> https://doi.org/10.3390/rs15010200

and link to this repository.

---

## Licence

MIT
