"""
src/crq/stats/surrogates_gpu.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GPU-accelerated (CuPy) surrogate generation for large-scale significance testing.

Provides batched phase-randomisation and IAAFT on an NVIDIA GPU.  Everything
stays on-device until the final ``return``, minimising host↔device transfers.

When CuPy / CUDA is unavailable the module falls back transparently to the
NumPy CPU implementations in ``surrogates.py``.

Architecture
------------
* ``phase_randomise_batch(X, seed)`` — X shape (n_surrogates, n_timesteps),
  all on GPU.  One cuFFT call covers the whole batch.
* ``iaaft_batch(X, seed, n_iter)`` — same shape convention; iterative,
  stays on GPU throughout.
* ``surrogate_xcorr_test_gpu(...)`` — drop-in replacement for the CPU
  ``surrogate_xcorr_test`` in surrogates.py; auto-dispatches to GPU or CPU.

Memory management
-----------------
For a batch of N surrogates of length T in float32:
  * raw data: N × T × 4 bytes
  * complex64 FFT: N × (T//2+1) × 8 bytes
  * IAAFT additionally needs sorted copy + ranks: N × T × 4 bytes each

``auto_batch_size(T, vram_budget_gb=10.0)`` computes the largest safe batch
that fits in VRAM with 20% headroom.

References
----------
Theiler et al. 1992, Schreiber & Schmitz 1996 (see surrogates.py for full refs).
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU availability probe
# ---------------------------------------------------------------------------

def _probe_gpu() -> tuple[bool, str]:
    """Return (available, reason_string)."""
    try:
        import cupy as cp                        # noqa: F401
        import cupy.fft as cpfft                 # noqa: F401
        _ = cp.array([1.0], dtype=cp.float32)    # force CUDA context init
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        raw_name = props["name"]
        name = raw_name.decode() if isinstance(raw_name, (bytes, bytearray)) else str(raw_name)
        total = props["totalGlobalMem"]
        logger.info("GPU: %s  VRAM: %.1f GB", name, total / 1e9)
        return True, f"{name} ({total/1e9:.1f} GB)"
    except ImportError:
        return False, "CuPy not installed"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


_GPU_AVAILABLE, _GPU_REASON = _probe_gpu()


def gpu_available() -> bool:
    return _GPU_AVAILABLE


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def auto_batch_size(
    n_timesteps: int,
    vram_budget_gb: float = 10.0,
    headroom: float = 0.20,
    dtype: str = "float32",
    method: str = "phase",
) -> int:
    """
    Return the largest batch size that fits in VRAM with *headroom* to spare.

    Memory breakdown per surrogate (float32):

    Phase:
      - signal tiled:    T × 4 bytes
      - sorted copy:     T × 4 bytes
      - rank buffer:     T × 4 bytes  (int32)
      - complex FFT:     (T//2+1) × 8 bytes
    Total ≈ T × 20 bytes

    IAAFT (method="iaaft"):
      - signal s:        T × 4 bytes
      - complex FFT:     (n_fft//2+1) × 8 bytes  (n_fft = next power-of-2)
      - argsort pass 1:  T × 8 bytes (int64 output) + ~1× thrust temp = T × 16
      - argsort pass 2:  T × 8 bytes (int64 output) + ~1× thrust temp = T × 16
    Total ≈ T × (4 + 8 + 32) = T × 44 bytes  (conservative — both argsort
    outputs live simultaneously in thrust's merge path)
    """
    bytes_per_elem = 4 if dtype == "float32" else 8
    T = n_timesteps
    if method == "iaaft":
        n_fft = _next_pow2(T)
        # IAAFT (chunked argsort, RANK_CHUNK=2048):
        #   signal s:    T × 4 bytes
        #   complex FFT: (n_fft//2+1) × 8 bytes  (padded to power-of-2)
        #   int32 ranks: T × 4 bytes  (reused each iteration)
        # Per-chunk argsort peak is amortised over n_surrogates (not counted here).
        bytes_per_surrogate = (
            T * bytes_per_elem               # signal s
            + (n_fft // 2 + 1) * 8          # complex64 FFT (padded)
            + T * 4                          # int32 rank buffer
        )
    else:
        # Phase: tiled signal + sorted copy + rank buffer + rfft
        bytes_per_surrogate = (
            T * bytes_per_elem           # signal
            + T * bytes_per_elem         # sorted copy
            + T * 4                      # int32 rank indices
            + (T // 2 + 1) * 8          # complex64 FFT
        )
    budget = vram_budget_gb * 1e9 * (1.0 - headroom)
    batch = max(1, int(budget // bytes_per_surrogate))
    logger.debug(
        "auto_batch_size: T=%d  bytes/surr=%.1f KB  budget=%.1f GB  batch=%d",
        T, bytes_per_surrogate / 1024, budget / 1e9, batch,
    )
    return batch


def _free_vram_bytes() -> int:
    """Return available VRAM bytes, or 0 if GPU unavailable."""
    if not _GPU_AVAILABLE:
        return 0
    try:
        import cupy as cp
        dev = cp.cuda.Device()
        return dev.mem_info[0]   # free bytes
    except Exception:
        return 0


def _free_memory_pool() -> None:
    """Release all CuPy memory-pool blocks back to the device."""
    if not _GPU_AVAILABLE:
        return
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GPU kernels — phase randomisation
# ---------------------------------------------------------------------------

def phase_randomise_batch_gpu(
    x: np.ndarray,
    n_surrogates: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate *n_surrogates* phase-randomised surrogates of 1-D signal *x*.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Input signal (CPU array).
    n_surrogates : int
        Number of surrogates to generate.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    surrogates : np.ndarray, shape (n_surrogates, T), float32
        CPU array of surrogate realisations.
    """
    import cupy as cp
    import cupy.fft as cpfft

    x = np.asarray(x, dtype=np.float32)
    T = len(x)
    n_rfft = T // 2 + 1
    rng = cp.random.default_rng(seed)

    # Upload and tile: shape (n_surrogates, T)
    x_gpu = cp.tile(cp.asarray(x), (n_surrogates, 1))

    # Forward rFFT: shape (n_surrogates, n_rfft) complex64
    X_gpu = cpfft.rfft(x_gpu, axis=1)

    # Random phases — uniform [0, 2π), shape (n_surrogates, n_rfft)
    phases = rng.uniform(0.0, 2.0 * math.pi, (n_surrogates, n_rfft)).astype(cp.float32)

    # Preserve DC and (if even) Nyquist
    phases[:, 0] = 0.0
    if T % 2 == 0:
        phases[:, -1] = 0.0

    # Apply phases: multiply by e^(iθ)
    X_surr = cp.abs(X_gpu) * cp.exp(1j * phases.astype(cp.complex64))

    # Restore DC and Nyquist exactly
    X_surr[:, 0] = X_gpu[:, 0]
    if T % 2 == 0:
        X_surr[:, -1] = X_gpu[:, -1]

    # Inverse rFFT
    out = cpfft.irfft(X_surr, n=T, axis=1)   # (n_surrogates, T) float32

    return cp.asnumpy(out)


# ---------------------------------------------------------------------------
# GPU kernels — IAAFT
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def iaaft_batch_gpu(
    x: np.ndarray,
    n_surrogates: int,
    seed: int | None = None,
    n_iter: int = 100,
) -> np.ndarray:
    """
    Generate *n_surrogates* IAAFT surrogates of 1-D signal *x* on GPU.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Input signal (CPU array).
    n_surrogates : int
    seed : int or None
    n_iter : int
        IAAFT iterations (default 100).

    Returns
    -------
    surrogates : np.ndarray, shape (n_surrogates, T), float32
    """
    import cupy as cp
    import cupy.fft as cpfft

    x = np.asarray(x, dtype=np.float32)
    T = len(x)
    n_fft = _next_pow2(T)
    rng = cp.random.default_rng(seed)

    x_gpu = cp.asarray(x)

    # Pre-compute target amplitude spectrum (padded)
    X_target = cpfft.rfft(x_gpu, n=n_fft)
    target_amp = cp.abs(X_target)                        # (n_fft//2+1,)

    # Sorted values of x (for rank-order rescaling) — broadcast over batch
    x_sorted = cp.sort(x_gpu)                            # (T,)

    # Initial surrogates: independent random permutations of x.
    # Process in row-chunks to keep peak VRAM low (avoids giant int64 argsort arrays).
    # RANK_CHUNK rows at a time: each chunk's argsort is B_chunk×T×8 bytes (int64) +
    # thrust temp, then we cast to int32 for the ranks output.
    RANK_CHUNK = min(n_surrogates, max(1, 2048))

    # Build s: chunk-by-chunk random permutations to limit peak VRAM
    s = cp.empty((n_surrogates, T), dtype=cp.float32)
    for i in range(0, n_surrogates, RANK_CHUNK):
        end = min(i + RANK_CHUNK, n_surrogates)
        keys_c = rng.random((end - i, T), dtype=cp.float32)
        perm_c = cp.argsort(keys_c, axis=1)
        s[i:end] = x_gpu[perm_c]
        del keys_c, perm_c

    # Pre-allocate int32 rank buffer (reused every iteration)
    ranks_i32 = cp.empty((n_surrogates, T), dtype=cp.int32)

    for _ in range(n_iter):
        # Step 1: adjust spectrum amplitude
        S = cpfft.rfft(s, n=n_fft, axis=1)              # (n_surrogates, n_fft//2+1)
        phase_s = cp.angle(S)
        del S
        S_adjusted = target_amp[cp.newaxis, :] * cp.exp(1j * phase_s.astype(cp.complex64))
        del phase_s
        s = cpfft.irfft(S_adjusted, n=T, axis=1)        # (n_surrogates, T) float32
        del S_adjusted

        # Step 2: rank-order rescaling — chunk-wise to bound argsort VRAM.
        # Each chunk: inner (B_chunk×T×8) + thrust_temp (~B_chunk×T×8) + ranks_chunk (int32).
        for i in range(0, n_surrogates, RANK_CHUNK):
            end = min(i + RANK_CHUNK, n_surrogates)
            inner_c = cp.argsort(s[i:end], axis=1)          # int64
            ranks_i32[i:end] = cp.argsort(inner_c, axis=1)  # int64 → int32
            del inner_c

        s = x_sorted[ranks_i32]                          # (n_surrogates, T) float32

    del ranks_i32
    return cp.asnumpy(s)


# ---------------------------------------------------------------------------
# Batched driver — handles VRAM limits automatically
# ---------------------------------------------------------------------------

def _run_batched(
    fn_gpu,          # phase_randomise_batch_gpu or iaaft_batch_gpu
    x: np.ndarray,
    n_surrogates: int,
    seed: int | None,
    batch_size: int,
    **kwargs,
) -> np.ndarray:
    """Split *n_surrogates* into batches of at most *batch_size*, run fn_gpu."""
    results = []
    rng_seeds = np.random.default_rng(seed).integers(0, 2**31, size=math.ceil(n_surrogates / batch_size))
    i = 0
    batch_idx = 0
    while i < n_surrogates:
        this_batch = min(batch_size, n_surrogates - i)
        chunk = fn_gpu(x, this_batch, seed=int(rng_seeds[batch_idx]), **kwargs)
        results.append(chunk)
        i += this_batch
        batch_idx += 1
        _free_memory_pool()   # return cached VRAM blocks between batches
    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Pearson correlation helper (CPU, vectorised — same as in surrogates.py)
# ---------------------------------------------------------------------------

def _pearson_lag_batch_cupy(
    surrogates: np.ndarray,   # (n_surr, T) float32, CPU array
    y: np.ndarray,            # (T,) float32, CPU array
    lag_bins: np.ndarray,     # (n_lags,) int
) -> np.ndarray:
    """
    GPU-accelerated batch Pearson r at all lags using CuPy cuBLAS.

    For each lag τ, the operation is a matrix–vector multiply:
        out[:, k] = (s_z[:, :n] @ y_z[lag:lag+n]) / n
    where s_z is z-scored surrogates, y_z is z-scored y, and n = T - |τ|.

    Returns float32 array (n_surr, n_lags) on CPU.
    """
    import cupy as cp

    n_surr, N = surrogates.shape
    n_lags = len(lag_bins)

    # Upload to GPU
    s_gpu = cp.asarray(surrogates)                        # (n_surr, N)
    y_gpu = cp.asarray(y)                                 # (N,)

    # z-score surrogates row-wise
    mu = s_gpu.mean(axis=1, keepdims=True)
    sd = s_gpu.std(axis=1, keepdims=True) + 1e-15
    s_z = (s_gpu - mu) / sd                               # (n_surr, N) float32

    # z-score y
    y_z = (y_gpu - y_gpu.mean()) / (y_gpu.std() + 1e-15)  # (N,) float32

    out = cp.empty((n_surr, n_lags), dtype=cp.float32)

    for k, lag in enumerate(lag_bins):
        if lag >= 0:
            n = N - lag
            if n > 1:
                # s_z[:, :n] @ y_z[lag:lag+n]  — cuBLAS SGEMV per lag
                out[:, k] = s_z[:, :n] @ y_z[lag: lag + n] / n
            else:
                out[:, k] = 0.0
        else:
            n = N + lag
            if n > 1:
                out[:, k] = s_z[:, -lag: -lag + n] @ y_z[:n] / n
            else:
                out[:, k] = 0.0

    return cp.asnumpy(out)


def _pearson_lag_array_cpu(x: np.ndarray, y: np.ndarray, lag_bins: np.ndarray) -> np.ndarray:
    """Vectorised Pearson r at each lag (CPU)."""
    N = len(x)
    x_z = (x - x.mean()) / (x.std() + 1e-15)
    y_z = (y - y.mean()) / (y.std() + 1e-15)
    rs = np.empty(len(lag_bins), dtype=np.float32)
    for k, lag in enumerate(lag_bins):
        if lag >= 0:
            n = N - lag
            rs[k] = np.dot(x_z[:n], y_z[lag: lag + n]) / n if n > 1 else 0.0
        else:
            n = N + lag
            rs[k] = np.dot(x_z[-lag: -lag + n], y_z[:n]) / n if n > 1 else 0.0
    return rs


def _pearson_lag_batch_gpu(
    surrogates: np.ndarray,   # (n_surr, T) float32
    y: np.ndarray,            # (T,) float32
    lag_bins: np.ndarray,     # (n_lags,) int
) -> np.ndarray:
    """
    Compute Pearson r(τ) between each row of *surrogates* and *y* at all lags.

    Uses CuPy cuBLAS SGEMV when the GPU is available (one matrix-vector
    multiply per lag — much faster than the serial CPU loop for large batches).
    Falls back to NumPy otherwise.

    Returns float32 array (n_surr, n_lags).
    """
    if _GPU_AVAILABLE:
        return _pearson_lag_batch_cupy(surrogates, y, lag_bins)

    N = surrogates.shape[1]
    # z-score each surrogate row
    mu = surrogates.mean(axis=1, keepdims=True)
    sd = surrogates.std(axis=1, keepdims=True) + 1e-15
    s_z = (surrogates - mu) / sd                         # (n_surr, T)

    y_z = (y - y.mean()) / (y.std() + 1e-15)            # (T,)

    n_surr, n_lags = len(surrogates), len(lag_bins)
    out = np.empty((n_surr, n_lags), dtype=np.float32)

    for k, lag in enumerate(lag_bins):
        if lag >= 0:
            n = N - lag
            if n > 1:
                out[:, k] = (s_z[:, :n] * y_z[lag: lag + n]).sum(axis=1) / n
            else:
                out[:, k] = 0.0
        else:
            n = N + lag
            if n > 1:
                out[:, k] = (s_z[:, -lag: -lag + n] * y_z[:n]).sum(axis=1) / n
            else:
                out[:, k] = 0.0
    return out


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for surrogate_xcorr_test
# ---------------------------------------------------------------------------

def surrogate_xcorr_test_gpu(
    x: np.ndarray,
    y: np.ndarray,
    lag_bins: np.ndarray,
    n_surrogates: int = 10_000,
    method: Literal["phase", "iaaft"] = "iaaft",
    seed: int | None = 42,
    iaaft_n_iter: int = 100,
    vram_budget_gb: float = 10.0,
    **_ignored,
) -> dict:
    """
    GPU-accelerated version of ``surrogate_xcorr_test``.

    Falls back to the CPU implementation if CUDA is unavailable.  The return
    dict has identical keys to the CPU version so callers need no changes.

    Parameters
    ----------
    x, y : array-like, shape (N,)
    lag_bins : array-like of int, lag offsets in bins
    n_surrogates : int
    method : "phase" | "iaaft"
    seed : int or None
    iaaft_n_iter : int
    vram_budget_gb : float
        Maximum VRAM to use (default 10 GB, leaving 2 GB for OS/driver).

    Returns
    -------
    dict — same keys as ``surrogate_xcorr_test``
    """
    if not _GPU_AVAILABLE:
        logger.warning("GPU unavailable (%s); falling back to CPU", _GPU_REASON)
        from crq.stats.surrogates import surrogate_xcorr_test
        return surrogate_xcorr_test(
            x, y, lag_bins,
            n_surrogates=n_surrogates,
            method=method,
            seed=seed,
            iaaft_n_iter=iaaft_n_iter,
        )

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    lag_bins = np.asarray(lag_bins, dtype=int)

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("NaN values in input arrays")

    # Release any cached VRAM from previous runs before sizing the batch
    _free_memory_pool()

    T = len(x)
    batch_size = auto_batch_size(T, vram_budget_gb=vram_budget_gb, method=method)
    logger.info(
        "GPU surrogate test (%s × %d): T=%d  batch_size=%d  device=%s",
        method, n_surrogates, T, batch_size, _GPU_REASON,
    )

    # Observed cross-correlation
    obs_r = _pearson_lag_array_cpu(x, y, lag_bins)
    obs_peak_idx = int(np.argmax(np.abs(obs_r)))
    obs_peak_r = float(obs_r[obs_peak_idx])
    obs_peak_lag = int(lag_bins[obs_peak_idx])

    logger.info(
        "Observed peak |r| = %.4f at lag bin %d", obs_peak_r, obs_peak_lag,
    )

    # Generate surrogates
    if method == "phase":
        gen_fn = phase_randomise_batch_gpu
        kwargs: dict = {}
    else:
        gen_fn = iaaft_batch_gpu
        kwargs = {"n_iter": iaaft_n_iter}

    surr_arrays = _run_batched(gen_fn, x, n_surrogates, seed, batch_size, **kwargs)
    # surr_arrays: (n_surrogates, T) float32

    # Cross-correlations for all surrogates
    surr_r = _pearson_lag_batch_gpu(surr_arrays, y, lag_bins)   # (n_surr, n_lags)

    # Global p-value: fraction of surrogates with max|r| ≥ observed max|r|
    surr_max_r = np.max(np.abs(surr_r), axis=1)                 # (n_surr,)
    p_global = float((surr_max_r >= abs(obs_peak_r)).mean())

    # Per-lag p-values
    p_at_lag = np.array(
        [(np.abs(surr_r[:, k]) >= abs(obs_r[k])).mean() for k in range(len(lag_bins))],
        dtype=np.float32,
    )

    logger.info(
        "p_global = %.4f  |  %d / %d surrogates exceeded peak",
        p_global, int((surr_max_r >= abs(obs_peak_r)).sum()), n_surrogates,
    )

    return {
        "observed_r":        obs_r.astype(np.float32),
        "observed_peak_r":   obs_peak_r,
        "observed_peak_lag": obs_peak_lag,
        "p_global":          p_global,
        "p_at_lag":          p_at_lag,
        "surrogate_r_arrays": surr_r.astype(np.float32),
        "surrogate_max_r":   surr_max_r.astype(np.float32),
        "n_surrogates":      n_surrogates,
        "method":            method,
        "seed":              seed,
    }
