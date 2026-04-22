#!/usr/bin/env python3
"""
scripts/catalog_completeness_check.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Plot and document USGS catalogue completeness as a function of magnitude
threshold and time.

Outputs
-------
results/catalog_completeness.png   — event count per year at 6 thresholds
results/catalog_completeness.txt   — text summary of effective start years

Usage
-----
python scripts/catalog_completeness_check.py
python scripts/catalog_completeness_check.py --data-dir /path/to/data/raw/usgs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crq.ingest.usgs import load_usgs

# Magnitude thresholds to test
THRESHOLDS = [4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
COLORS      = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3", "#a65628"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "usgs")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--start-year", type=int, default=1960)
    p.add_argument("--end-year",   type=int, default=2019)
    return p.parse_args()


def _load_events(data_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    return load_usgs(start_year, end_year, data_dir)


def _annual_counts(events: pd.DataFrame, threshold: float) -> pd.Series:
    """Events per year at or above *threshold*."""
    filtered = events[events["mag"] >= threshold]
    return filtered.groupby(filtered.index.year).size()


def _stability_year(annual: pd.Series, window: int = 5, cv_max: float = 0.15) -> int | None:
    """
    Estimate the first year after which the annual event count stabilises.

    Uses a sliding window coefficient of variation (std/mean).  Returns the
    first year where the CV of the subsequent *window*-year block falls below
    *cv_max*, or None if the series never stabilises.
    """
    years = annual.index.tolist()
    values = annual.values.astype(float)
    for i in range(len(years) - window):
        block = values[i : i + window]
        if np.mean(block) > 0 and np.std(block) / np.mean(block) < cv_max:
            return years[i]
    return None


def _gutenberg_richter(events: pd.DataFrame, mag_min: float = 4.5, mag_max: float = 9.5) -> tuple[float, float]:
    """
    Fit Gutenberg-Richter b-value using maximum likelihood (Aki 1965).

    Returns (b, a) for log10(N) = a - b·M.
    """
    mags = events["mag"].dropna()
    mags = mags[(mags >= mag_min) & (mags <= mag_max)]
    if len(mags) < 10:
        return np.nan, np.nan
    b = np.log10(np.e) / (mags.mean() - mag_min)
    a = np.log10(len(mags)) + b * mag_min
    return float(b), float(a)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading USGS events from {args.data_dir} …")
    events = _load_events(args.data_dir, args.start_year, args.end_year)
    if events.empty:
        print("ERROR: no events loaded — run 01_download_data.py first")
        sys.exit(1)
    print(f"  {len(events):,} events loaded, spanning "
          f"{events.index.min().year}–{events.index.max().year}")

    # ----------------------------------------------------------------
    # Figure 1: annual event counts per magnitude threshold
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                              gridspec_kw={"height_ratios": [3, 1]})
    ax_count, ax_ratio = axes

    year_range = np.arange(args.start_year, args.end_year + 1)
    summary_lines: list[str] = []
    annual_all: dict[float, pd.Series] = {}

    for thresh, color in zip(THRESHOLDS, COLORS):
        annual = _annual_counts(events, thresh).reindex(year_range, fill_value=0)
        annual_all[thresh] = annual
        ax_count.plot(annual.index, annual.values, color=color,
                      label=f"M ≥ {thresh:.1f}", linewidth=1.5 if thresh != 4.5 else 2.5)
        stab = _stability_year(annual)
        marker = f"stable from ~{stab}" if stab else "no clear stabilisation"
        summary_lines.append(f"M ≥ {thresh:.1f}: {marker}")

    ax_count.set_ylabel("Events per year")
    ax_count.set_title("USGS Earthquake Catalogue — Annual Event Counts by Magnitude Threshold")
    ax_count.legend(loc="upper left")
    ax_count.grid(True, alpha=0.3)
    ax_count.set_xlim(args.start_year, args.end_year)

    # Panel 2: ratio M≥4.5 / M≥5.0 — should be roughly constant when both complete
    r45 = annual_all[4.5]
    r50 = annual_all[5.0]
    ratio = (r45 / r50.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ax_ratio.plot(ratio.index, ratio.values, color="black", linewidth=1.0)
    ax_ratio.axhline(ratio.loc[1990:2019].median(), color="red", linestyle="--",
                     label=f"Median (1990-2019): {ratio.loc[1990:2019].median():.1f}")
    ax_ratio.set_ylabel("Ratio M≥4.5 / M≥5.0")
    ax_ratio.set_xlabel("Year")
    ax_ratio.legend()
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.set_xlim(args.start_year, args.end_year)

    fig.tight_layout()
    plot_path = args.output_dir / "catalog_completeness.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")

    # ----------------------------------------------------------------
    # Figure 2: Gutenberg-Richter frequency-magnitude diagram
    # ----------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    mag_bins = np.arange(4.0, 9.5, 0.2)
    counts_ge = [(events["mag"] >= m).sum() for m in mag_bins]
    ax2.semilogy(mag_bins, counts_ge, "ko", markersize=4, label="Observed N(≥M)")

    b, a = _gutenberg_richter(events)
    if not np.isnan(b):
        m_fit = np.linspace(4.5, 9.0, 100)
        n_fit = 10 ** (a - b * m_fit)
        ax2.semilogy(m_fit, n_fit, "r-", label=f"G-R fit: b={b:.2f}")

    ax2.set_xlabel("Magnitude M")
    ax2.set_ylabel("Cumulative count N(≥M)")
    ax2.set_title("Gutenberg-Richter Frequency-Magnitude Distribution")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    gr_path = args.output_dir / "gutenberg_richter.png"
    fig2.savefig(gr_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {gr_path}")

    # ----------------------------------------------------------------
    # Text summary
    # ----------------------------------------------------------------
    txt_path = args.output_dir / "catalog_completeness.txt"
    lines = [
        "USGS Earthquake Catalogue — Completeness Summary",
        "=" * 50,
        f"Total events loaded: {len(events):,}",
        f"Date range: {events.index.min().date()} — {events.index.max().date()}",
        f"Gutenberg-Richter b-value (M≥4.5): {b:.3f}" if not np.isnan(b) else "G-R fit: insufficient data",
        "",
        "Estimated catalogue completeness onset (CV < 15% over 5-year window):",
    ] + [f"  {s}" for s in summary_lines] + [
        "",
        "NOTE: Use M≥4.5 from ~1976 onwards for global cross-correlation analysis.",
        "Earlier data should be treated with caution — counts are systematically low.",
    ]
    txt_path.write_text("\n".join(lines))
    print(f"Saved: {txt_path}")
    print()
    for line in lines:
        print(line)


if __name__ == "__main__":
    run(_parse_args())
