# Cosmic Ray / Earthquake Correlation Study

## Scientific context
Testing Homola et al. 2023 (JASTP 247, 106068, DOI 10.1016/j.jastp.2023.106068) 
which claims a 6σ correlation between cosmic ray rate variations and global seismic 
activity at a 15-day lag, with ~11-year periodicity. Prior replication attempt 
using independent NMDB data failed. This project is a rigorous methodological 
replication + stress test.

## Key hypotheses to test
1. The 15-day lag signal survives phase-randomised surrogate null models
2. The signal survives solar cycle detrending (both series share an ~11-year component)
3. The signal is stable across NMDB station subsets
4. The signal is stable across time windows (e.g., 1980-95, 1995-2010, 2005-19)
5. The signal appears with seismic moment (M₀ ∝ 10^(1.5Mw)) not raw magnitude sum

## Data sources
- NMDB: http://nest.nmdb.eu (44 stations, pressure-corrected hourly neutron counts)
- USGS: earthquake catalogue (M≥4.5 for completeness)
- SIDC/KSO: daily sunspot numbers
- Reference: Pierre Auger scaler data (optional, secondary)

## Hardware
- CPU: Xeon E5-2680 v4 (14 physical / 28 logical cores)
- RAM: 60 GB
- GPU: Nvidia M40, 12 GB VRAM, compute capability 5.2 (CUDA 11.x max)
- Prefer CuPy over PyTorch for FFT work; M40 does not support bfloat16

## Code standards
- Python 3.11+, type hints required for public functions
- pytest for all data-transformation functions
- Keep raw data in `data/raw/`, processed in `data/processed/`, results in `results/`
- All station metadata (lat, lon, altitude, rigidity cutoff) in `config/stations.yaml`
- Seed all random operations; log seed + git SHA in every results file
- Use `polars` or `pandas` with `pyarrow` backend (not default pandas) for speed
