#!/usr/bin/env python3
"""
scripts/benchmark_gpu.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Benchmark CPU (joblib, 28 cores) vs GPU (CuPy) surrogate generation.

Test signal: N=16,000-point AR(1) with ρ=0.85 (pathologically bad FFT size
for CPU — 16,000 = 2^7 × 5^4, actually a reasonable FFT size, chosen to be
larger than the real 3,215-point series but still manageable on GPU).

n_surrogates = 10,000  (configurable via --n-surrogates)

Validates numerical equivalence: CPU vs GPU surrogate p-values must agree to
within ±2/sqrt(n_surrogates) Monte Carlo error.

Outputs
-------
results/benchmark_gpu.txt   — timing table and equivalence check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crq.stats.surrogates import surrogate_xcorr_test
from crq.stats.surrogates_gpu import (
    gpu_available,
    surrogate_xcorr_test_gpu,
    auto_batch_size,
    _GPU_REASON,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ar1(N: int, rho: float = 0.85, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.empty(N, dtype=np.float32)
    y = np.empty(N, dtype=np.float32)
    x[0] = y[0] = rng.standard_normal()
    noise = rng.standard_normal((2, N)).astype(np.float32)
    for i in range(1, N):
        x[i] = rho * x[i-1] + np.sqrt(1 - rho**2) * noise[0, i]
        y[i] = rho * y[i-1] + np.sqrt(1 - rho**2) * noise[1, i]
    return x, y


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}min"


def _print_banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-surrogates", type=int, default=10_000)
    p.add_argument("--n-timesteps",  type=int, default=16_000)
    p.add_argument("--method",       default="both", choices=["phase", "iaaft", "both"])
    p.add_argument("--iaaft-iter",   type=int, default=100)
    p.add_argument("--n-jobs",       type=int, default=-1,  help="CPU joblib workers")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--lag-range",    type=int, default=200, help="±lag range in bins")
    p.add_argument("--vram-gb",      type=float, default=10.0)
    p.add_argument("--output-dir",   type=Path,  default=PROJECT_ROOT / "results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Run one method, CPU then GPU, return timing + equivalence result
# ---------------------------------------------------------------------------

def run_comparison(
    x: np.ndarray,
    y: np.ndarray,
    lag_bins: np.ndarray,
    method: str,
    n_surrogates: int,
    n_jobs: int,
    iaaft_iter: int,
    seed: int,
    vram_gb: float,
) -> dict:
    _print_banner(f"Method: {method.upper()}  |  N={len(x)}  |  n_surr={n_surrogates:,}")

    result = {}

    # ------------------------------------------------------------------
    # CPU benchmark
    # ------------------------------------------------------------------
    print(f"\n[CPU] joblib n_jobs={n_jobs}, method={method}")
    t0 = time.perf_counter()
    cpu_out = surrogate_xcorr_test(
        x, y, lag_bins,
        n_surrogates=n_surrogates,
        method=method,
        seed=seed,
        n_jobs=n_jobs,
        iaaft_n_iter=iaaft_iter,
    )
    cpu_time = time.perf_counter() - t0
    print(f"  p_global = {cpu_out['p_global']:.4f}  |  time = {_fmt_time(cpu_time)}")
    result["cpu_time_s"]    = cpu_time
    result["cpu_p_global"]  = cpu_out["p_global"]

    # ------------------------------------------------------------------
    # GPU benchmark
    # ------------------------------------------------------------------
    if not gpu_available():
        print(f"\n[GPU] UNAVAILABLE — {_GPU_REASON}")
        result["gpu_time_s"]   = None
        result["gpu_p_global"] = None
        result["speedup"]      = None
        result["equiv_ok"]     = None
        return result

    T = len(x)
    batch = auto_batch_size(T, vram_budget_gb=vram_gb, method=method)
    print(f"\n[GPU] device={_GPU_REASON}  batch_size={batch:,}")

    # Warm-up pass (avoids counting CUDA JIT in timing)
    _ = surrogate_xcorr_test_gpu(
        x, y, lag_bins[:10],
        n_surrogates=32,
        method=method,
        seed=seed,
        iaaft_n_iter=min(iaaft_iter, 5),
        vram_budget_gb=vram_gb,
    )

    t0 = time.perf_counter()
    gpu_out = surrogate_xcorr_test_gpu(
        x, y, lag_bins,
        n_surrogates=n_surrogates,
        method=method,
        seed=seed,
        iaaft_n_iter=iaaft_iter,
        vram_budget_gb=vram_gb,
    )
    gpu_time = time.perf_counter() - t0
    speedup = cpu_time / gpu_time
    print(f"  p_global = {gpu_out['p_global']:.4f}  |  time = {_fmt_time(gpu_time)}  |  speedup = {speedup:.1f}×")

    # ------------------------------------------------------------------
    # Numerical equivalence check
    # ±2/sqrt(n_surrogates) Monte Carlo tolerance on p_global
    # ------------------------------------------------------------------
    mc_tol = 2.0 / np.sqrt(n_surrogates)
    delta = abs(cpu_out["p_global"] - gpu_out["p_global"])
    equiv_ok = delta <= mc_tol
    status = "PASS ✓" if equiv_ok else "FAIL ✗"
    print(f"\n  Equivalence check: |Δp_global| = {delta:.4f}  (tolerance {mc_tol:.4f})  → {status}")

    # Also check peak lag agreement
    cpu_peak = cpu_out["observed_peak_lag"]
    gpu_peak = gpu_out["observed_peak_lag"]
    if cpu_peak == gpu_peak:
        print(f"  Peak lag:  CPU={cpu_peak}  GPU={gpu_peak}  → AGREE ✓")
    else:
        print(f"  Peak lag:  CPU={cpu_peak}  GPU={gpu_peak}  → DIFFER (check seed handling)")

    result["gpu_time_s"]   = gpu_time
    result["gpu_p_global"] = gpu_out["p_global"]
    result["speedup"]      = speedup
    result["equiv_ok"]     = equiv_ok
    result["mc_tol"]       = mc_tol
    result["delta_p"]      = delta
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding AR(1) series: N={args.n_timesteps:,}  ρ=0.85  seed={args.seed}")
    x, y = make_ar1(args.n_timesteps, seed=args.seed)
    lag_bins = np.arange(-args.lag_range, args.lag_range + 1, dtype=int)

    if gpu_available():
        print(f"GPU: {_GPU_REASON}")
        for m in (["phase", "iaaft"] if args.method == "both" else [args.method]):
            batch = auto_batch_size(args.n_timesteps, vram_budget_gb=args.vram_gb, method=m)
            print(f"Auto batch size for T={args.n_timesteps} ({m}): {batch:,} surrogates/pass")
    else:
        print(f"GPU: UNAVAILABLE — {_GPU_REASON}  (running CPU-only benchmark)")

    methods = ["phase", "iaaft"] if args.method == "both" else [args.method]
    all_results = {}

    for m in methods:
        r = run_comparison(
            x, y, lag_bins,
            method=m,
            n_surrogates=args.n_surrogates,
            n_jobs=args.n_jobs,
            iaaft_iter=args.iaaft_iter,
            seed=args.seed,
            vram_gb=args.vram_gb,
        )
        all_results[m] = r

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    _print_banner("Summary")
    print(f"\n{'Method':<8} {'CPU (s)':>10} {'GPU (s)':>10} {'Speedup':>9} {'Equiv':>7}")
    print("-" * 48)
    for m, r in all_results.items():
        gpu_s   = f"{r['gpu_time_s']:.1f}" if r["gpu_time_s"] is not None else "N/A"
        speedup = f"{r['speedup']:.1f}×"   if r["speedup"]   is not None else "N/A"
        equiv   = ("PASS" if r["equiv_ok"] else "FAIL") if r["equiv_ok"] is not None else "N/A"
        print(f"{m:<8} {r['cpu_time_s']:>10.1f} {gpu_s:>10} {speedup:>9} {equiv:>7}")

    # ------------------------------------------------------------------
    # Save text report
    # ------------------------------------------------------------------
    lines = [
        "GPU vs CPU Surrogate Benchmark",
        "=" * 50,
        f"N timesteps:   {args.n_timesteps:,}",
        f"N surrogates:  {args.n_surrogates:,}",
        f"IAAFT iter:    {args.iaaft_iter}",
        f"CPU n_jobs:    {args.n_jobs}",
        f"GPU device:    {_GPU_REASON}",
        "",
        f"{'Method':<8} {'CPU (s)':>10} {'GPU (s)':>10} {'Speedup':>9} {'Equiv':>7}",
        "-" * 48,
    ]
    for m, r in all_results.items():
        gpu_s   = f"{r['gpu_time_s']:.1f}" if r["gpu_time_s"] is not None else "N/A"
        speedup = f"{r['speedup']:.1f}x"   if r["speedup"]   is not None else "N/A"
        equiv   = ("PASS" if r["equiv_ok"] else "FAIL") if r["equiv_ok"] is not None else "N/A"
        lines.append(f"{m:<8} {r['cpu_time_s']:>10.1f} {gpu_s:>10} {speedup:>9} {equiv:>7}")

    out_path = args.output_dir / "benchmark_gpu.txt"
    out_path.write_text("\n".join(lines))
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    run(_parse_args())
