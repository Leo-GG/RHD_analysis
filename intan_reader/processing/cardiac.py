"""
Cardiac field potential analysis for MEA recordings.

Provides methods for estimating field potential duration (FPD), which is
analogous to the QT interval in ECG recordings. Suitable for LFP recordings
from cardiomyocytes on multi-electrode arrays.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as _sig
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def estimate_qt_interval(
    waveform: np.ndarray,
    sample_rate: float,
    *,
    t_search_start_ms: float = 50.0,
    t_search_end_ms: float = 500.0,
    derivative_threshold: float = 0.1,
    baseline_threshold: float = 0.1,
    smoothing_ms: float = 5.0,
) -> Dict[str, any]:
    """Estimate QT interval (field potential duration) from a single waveform.

    Uses three methods:
    1. **Derivative method**: T-end where smoothed derivative returns to near-zero
    2. **Baseline crossing method**: T-end where signal returns to baseline level
    3. **Tangent method**: T-end where tangent at steepest descent intersects baseline

    Parameters
    ----------
    waveform : np.ndarray
        1-D waveform snippet centered on the depolarization peak.
    sample_rate : float
        Sampling rate in Hz.
    t_search_start_ms : float, optional
        Start searching for T-end this many ms after Q point. Default 50 ms.
    t_search_end_ms : float, optional
        Stop searching for T-end this many ms after Q point. Default 500 ms.
    derivative_threshold : float, optional
        Fraction of max derivative magnitude to consider as "near zero".
        Default 0.1 (10% of peak derivative).
    baseline_threshold : float, optional
        Fraction of T-wave amplitude to consider as "at baseline".
        Default 0.1 (10% of T-wave amplitude).
    smoothing_ms : float, optional
        Smoothing window for derivative in ms. Default 5 ms.

    Returns
    -------
    dict
        Dictionary with keys:

        - **q_idx** : sample index of Q point (minimum)
        - **t_peak_idx** : sample index of T-wave peak
        - **qt_deriv_ms** : QT from derivative method (ms), NaN if failed
        - **t_end_deriv_idx** : T-end index from derivative method, -1 if failed
        - **qt_baseline_ms** : QT from baseline crossing method (ms), NaN if failed
        - **t_end_baseline_idx** : T-end index from baseline method, -1 if failed
        - **qt_tangent_ms** : QT from tangent method (ms), NaN if failed
        - **t_end_tangent_idx** : T-end index from tangent method, -1 if failed
        - **tangent_slope** : slope of tangent line (for visualization)
        - **tangent_intercept** : intercept of tangent line (for visualization)
        - **qt_gauss_ms** : QT from Gaussian fit method (ms), NaN if failed
        - **t_end_gauss_idx** : T-end index from Gaussian method, -1 if failed
        - **gauss_params** : Gaussian fit parameters (amplitude, mu, sigma, offset)
        - **qt_avg_ms** : average of all valid methods (ms), NaN if none valid
    """
    n_samples = len(waveform)
    samples_per_ms = sample_rate / 1000.0

    # Find Q point (minimum, most negative deflection)
    q_idx = int(np.argmin(waveform))

    # Define search window for T-end
    t_start_idx = q_idx + int(t_search_start_ms * samples_per_ms)
    t_end_search_idx = min(n_samples, q_idx + int(t_search_end_ms * samples_per_ms))

    result = {
        "q_idx": q_idx,
        "t_peak_idx": -1,
        "qt_deriv_ms": np.nan,
        "t_end_deriv_idx": -1,
        "qt_baseline_ms": np.nan,
        "t_end_baseline_idx": -1,
        "qt_tangent_ms": np.nan,
        "t_end_tangent_idx": -1,
        "tangent_slope": np.nan,
        "tangent_intercept": np.nan,
        "qt_gauss_ms": np.nan,
        "t_end_gauss_idx": -1,
        "gauss_params": None,
        "qt_avg_ms": np.nan,
    }

    if t_start_idx >= t_end_search_idx or t_start_idx >= n_samples:
        return result

    # Extract search region
    search_region = waveform[t_start_idx:t_end_search_idx]

    # Compute smoothed derivative
    smoothing_samples = max(3, int(smoothing_ms * samples_per_ms))
    if smoothing_samples % 2 == 0:
        smoothing_samples += 1

    # Smooth the search region for peak detection
    search_smooth = uniform_filter1d(search_region, size=smoothing_samples, mode="reflect")

    # Find T-wave peak (local maximum) in the search region
    t_peak_local = int(np.argmax(search_smooth))
    result["t_peak_idx"] = t_start_idx + t_peak_local

    # Compute baseline from end of search region
    baseline_samples = max(int(10 * samples_per_ms), 1)
    baseline = np.mean(search_region[-baseline_samples:])

    # =========================================================================
    # METHOD 1: Derivative threshold method
    # =========================================================================
    post_peak_start = t_peak_local + int(5 * samples_per_ms)
    if post_peak_start < len(search_region):
        post_peak_region = search_region[post_peak_start:]

        deriv = np.gradient(post_peak_region)
        deriv_smooth = uniform_filter1d(deriv, size=smoothing_samples, mode="reflect")

        max_deriv = np.max(np.abs(deriv_smooth))
        if max_deriv < 1e-10:
            t_end_deriv_local = post_peak_start
        else:
            threshold = derivative_threshold * max_deriv
            below_threshold = np.abs(deriv_smooth) < threshold

            min_sustained = max(5, int(2.0 * samples_per_ms))
            t_end_deriv_local = -1

            for i in range(len(below_threshold) - min_sustained):
                if np.all(below_threshold[i:i + min_sustained]):
                    t_end_deriv_local = post_peak_start + i
                    break

            if t_end_deriv_local < 0:
                # Fallback for derivative method
                distances = np.abs(post_peak_region - baseline)
                close_to_baseline = distances < 0.2 * np.max(np.abs(search_region - baseline))
                indices = np.where(close_to_baseline)[0]
                if len(indices) > 0:
                    t_end_deriv_local = post_peak_start + indices[0]

        if t_end_deriv_local >= 0:
            t_end_deriv_idx = t_start_idx + t_end_deriv_local
            result["t_end_deriv_idx"] = t_end_deriv_idx
            result["qt_deriv_ms"] = (t_end_deriv_idx - q_idx) / samples_per_ms

    # =========================================================================
    # METHOD 2: Baseline crossing method
    # =========================================================================
    # Find where signal crosses back to baseline after T-wave peak
    t_peak_value = search_smooth[t_peak_local]
    t_wave_amplitude = abs(t_peak_value - baseline)

    if t_wave_amplitude > 1e-10 and post_peak_start < len(search_region):
        post_peak_region = search_region[post_peak_start:]
        post_peak_smooth = uniform_filter1d(post_peak_region, size=smoothing_samples, mode="reflect")

        # Find where signal returns within threshold of baseline
        baseline_band = baseline_threshold * t_wave_amplitude
        within_baseline = np.abs(post_peak_smooth - baseline) < baseline_band

        # Find first sustained crossing
        min_sustained = max(5, int(2.0 * samples_per_ms))
        t_end_baseline_local = -1

        for i in range(len(within_baseline) - min_sustained):
            if np.all(within_baseline[i:i + min_sustained]):
                t_end_baseline_local = post_peak_start + i
                break

        if t_end_baseline_local < 0:
            # Fallback: find first crossing point
            indices = np.where(within_baseline)[0]
            if len(indices) > 0:
                t_end_baseline_local = post_peak_start + indices[0]

        if t_end_baseline_local >= 0:
            t_end_baseline_idx = t_start_idx + t_end_baseline_local
            result["t_end_baseline_idx"] = t_end_baseline_idx
            result["qt_baseline_ms"] = (t_end_baseline_idx - q_idx) / samples_per_ms

    # =========================================================================
    # METHOD 3: Tangent method (steepest descent intersects baseline)
    # =========================================================================
    # Find the point of steepest descent on the T-wave (most negative derivative)
    if post_peak_start < len(search_region):
        post_peak_region = search_region[post_peak_start:]
        deriv = np.gradient(post_peak_region)
        deriv_smooth = uniform_filter1d(deriv, size=smoothing_samples, mode="reflect")

        # Find steepest descent (most negative derivative)
        steepest_idx_local = int(np.argmin(deriv_smooth))
        steepest_idx_in_search = post_peak_start + steepest_idx_local

        if steepest_idx_in_search < len(search_region) - 1:
            # Get the slope at steepest point (in amplitude units per sample)
            slope = deriv_smooth[steepest_idx_local]

            if abs(slope) > 1e-10:
                # Get the value at steepest point
                steepest_value = search_region[steepest_idx_in_search]

                # Tangent line: y = slope * (x - steepest_idx) + steepest_value
                # Find where tangent crosses baseline: baseline = slope * (x - steepest_idx) + steepest_value
                # x = steepest_idx + (baseline - steepest_value) / slope
                x_intercept = steepest_idx_in_search + (baseline - steepest_value) / slope

                # Store tangent parameters for visualization (in absolute indices)
                result["tangent_slope"] = slope
                result["tangent_intercept"] = steepest_value - slope * steepest_idx_in_search

                # Check if intercept is within valid range
                if 0 <= x_intercept < len(search_region):
                    t_end_tangent_idx = t_start_idx + int(x_intercept)
                    if t_end_tangent_idx > q_idx:
                        result["t_end_tangent_idx"] = t_end_tangent_idx
                        result["qt_tangent_ms"] = (t_end_tangent_idx - q_idx) / samples_per_ms

    # =========================================================================
    # METHOD 4: Gaussian fit method
    # =========================================================================
    # Fit a Gaussian to the T-wave and define T-end as mu + 2*sigma (95% of area)
    def gaussian(x, amplitude, mu, sigma, offset):
        return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

    try:
        # Use the search region for fitting
        x_data = np.arange(len(search_region))
        y_data = search_region

        # Initial guesses
        amp_guess = search_smooth[t_peak_local] - baseline
        mu_guess = t_peak_local
        sigma_guess = len(search_region) / 6  # Rough estimate
        offset_guess = baseline

        # Bounds to keep fit reasonable
        bounds = (
            [0, 0, 1, -np.inf],  # Lower bounds
            [np.inf, len(search_region), len(search_region) / 2, np.inf]  # Upper bounds
        )

        popt, _ = curve_fit(
            gaussian, x_data, y_data,
            p0=[amp_guess, mu_guess, sigma_guess, offset_guess],
            bounds=bounds,
            maxfev=1000
        )

        amplitude, mu, sigma, offset = popt
        result["gauss_params"] = {
            "amplitude": float(amplitude),
            "mu": float(mu),
            "sigma": float(sigma),
            "offset": float(offset),
        }

        # T-end defined as mu + 2*sigma (covers ~95% of Gaussian)
        t_end_gauss_local = mu + 2 * sigma

        if 0 <= t_end_gauss_local < len(search_region):
            t_end_gauss_idx = t_start_idx + int(t_end_gauss_local)
            if t_end_gauss_idx > q_idx:
                result["t_end_gauss_idx"] = t_end_gauss_idx
                result["qt_gauss_ms"] = (t_end_gauss_idx - q_idx) / samples_per_ms

    except (RuntimeError, ValueError):
        # Fit failed - leave as NaN
        pass

    # =========================================================================
    # Compute average of all valid methods
    # =========================================================================
    qt_values = []
    if not np.isnan(result["qt_deriv_ms"]):
        qt_values.append(result["qt_deriv_ms"])
    if not np.isnan(result["qt_baseline_ms"]):
        qt_values.append(result["qt_baseline_ms"])
    if not np.isnan(result["qt_tangent_ms"]):
        qt_values.append(result["qt_tangent_ms"])
    if not np.isnan(result["qt_gauss_ms"]):
        qt_values.append(result["qt_gauss_ms"])

    if qt_values:
        result["qt_avg_ms"] = float(np.mean(qt_values))

    return result


def compute_qt_intervals(
    waveforms: Dict[int, List[np.ndarray]],
    sample_rate: float,
    *,
    exclude_channels: Optional[List[int]] = None,
    t_search_start_ms: float = 50.0,
    t_search_end_ms: float = 500.0,
    derivative_threshold: float = 0.1,
    baseline_threshold: float = 0.1,
    smoothing_ms: float = 5.0,
) -> "pd.DataFrame":
    """Compute QT intervals for all waveforms across channels.

    Uses four methods and reports statistics for each plus their average.

    Parameters
    ----------
    waveforms : dict[int, list of np.ndarray]
        Per-channel waveform snippets (from :func:`extract_waveforms`).
    sample_rate : float
        Sampling rate in Hz.
    exclude_channels : list of int, optional
        Channels to exclude (e.g., noisy channels).
    t_search_start_ms : float, optional
        Start searching for T-end this many ms after Q point. Default 50 ms.
    t_search_end_ms : float, optional
        Stop searching for T-end this many ms after Q point. Default 500 ms.
    derivative_threshold : float, optional
        Fraction of max derivative magnitude for T-end detection. Default 0.1.
    baseline_threshold : float, optional
        Fraction of T-wave amplitude for baseline crossing detection. Default 0.1.
    smoothing_ms : float, optional
        Smoothing window for derivative in ms. Default 5 ms.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for each method (deriv, baseline, tangent, gauss, avg):

        - **channel** : channel index
        - **n_beats_deriv/baseline/tangent/gauss/avg** : valid measurements per method
        - **qt_deriv_min/max/mean/median/std_ms** : derivative method stats
        - **qt_baseline_min/max/mean/median/std_ms** : baseline method stats
        - **qt_tangent_min/max/mean/median/std_ms** : tangent method stats
        - **qt_gauss_min/max/mean/median/std_ms** : Gaussian fit method stats
        - **qt_avg_min/max/mean/median/std_ms** : average of all methods stats
    """
    import pandas as pd

    exclude_set = set(exclude_channels) if exclude_channels else set()

    rows = []
    for ch in sorted(waveforms.keys()):
        if ch in exclude_set:
            continue

        waves = waveforms[ch]
        if not waves:
            continue

        qt_deriv_values = []
        qt_baseline_values = []
        qt_tangent_values = []
        qt_gauss_values = []
        qt_avg_values = []

        for w in waves:
            result = estimate_qt_interval(
                w,
                sample_rate,
                t_search_start_ms=t_search_start_ms,
                t_search_end_ms=t_search_end_ms,
                derivative_threshold=derivative_threshold,
                baseline_threshold=baseline_threshold,
                smoothing_ms=smoothing_ms,
            )

            if not np.isnan(result["qt_deriv_ms"]):
                qt_deriv_values.append(result["qt_deriv_ms"])

            if not np.isnan(result["qt_baseline_ms"]):
                qt_baseline_values.append(result["qt_baseline_ms"])

            if not np.isnan(result["qt_tangent_ms"]):
                qt_tangent_values.append(result["qt_tangent_ms"])

            if not np.isnan(result["qt_gauss_ms"]):
                qt_gauss_values.append(result["qt_gauss_ms"])

            if not np.isnan(result["qt_avg_ms"]):
                qt_avg_values.append(result["qt_avg_ms"])

        # Skip channel if no valid measurements from any method
        if not qt_deriv_values and not qt_baseline_values and not qt_tangent_values and not qt_gauss_values:
            continue

        row = {"channel": ch}

        # Derivative method stats
        if qt_deriv_values:
            arr = np.array(qt_deriv_values)
            row["n_beats_deriv"] = len(arr)
            row["qt_deriv_min_ms"] = float(np.min(arr))
            row["qt_deriv_max_ms"] = float(np.max(arr))
            row["qt_deriv_mean_ms"] = float(np.mean(arr))
            row["qt_deriv_median_ms"] = float(np.median(arr))
            row["qt_deriv_std_ms"] = float(np.std(arr))
        else:
            row["n_beats_deriv"] = 0
            row["qt_deriv_min_ms"] = np.nan
            row["qt_deriv_max_ms"] = np.nan
            row["qt_deriv_mean_ms"] = np.nan
            row["qt_deriv_median_ms"] = np.nan
            row["qt_deriv_std_ms"] = np.nan

        # Baseline method stats
        if qt_baseline_values:
            arr = np.array(qt_baseline_values)
            row["n_beats_baseline"] = len(arr)
            row["qt_baseline_min_ms"] = float(np.min(arr))
            row["qt_baseline_max_ms"] = float(np.max(arr))
            row["qt_baseline_mean_ms"] = float(np.mean(arr))
            row["qt_baseline_median_ms"] = float(np.median(arr))
            row["qt_baseline_std_ms"] = float(np.std(arr))
        else:
            row["n_beats_baseline"] = 0
            row["qt_baseline_min_ms"] = np.nan
            row["qt_baseline_max_ms"] = np.nan
            row["qt_baseline_mean_ms"] = np.nan
            row["qt_baseline_median_ms"] = np.nan
            row["qt_baseline_std_ms"] = np.nan

        # Tangent method stats
        if qt_tangent_values:
            arr = np.array(qt_tangent_values)
            row["n_beats_tangent"] = len(arr)
            row["qt_tangent_min_ms"] = float(np.min(arr))
            row["qt_tangent_max_ms"] = float(np.max(arr))
            row["qt_tangent_mean_ms"] = float(np.mean(arr))
            row["qt_tangent_median_ms"] = float(np.median(arr))
            row["qt_tangent_std_ms"] = float(np.std(arr))
        else:
            row["n_beats_tangent"] = 0
            row["qt_tangent_min_ms"] = np.nan
            row["qt_tangent_max_ms"] = np.nan
            row["qt_tangent_mean_ms"] = np.nan
            row["qt_tangent_median_ms"] = np.nan
            row["qt_tangent_std_ms"] = np.nan

        # Gaussian method stats
        if qt_gauss_values:
            arr = np.array(qt_gauss_values)
            row["n_beats_gauss"] = len(arr)
            row["qt_gauss_min_ms"] = float(np.min(arr))
            row["qt_gauss_max_ms"] = float(np.max(arr))
            row["qt_gauss_mean_ms"] = float(np.mean(arr))
            row["qt_gauss_median_ms"] = float(np.median(arr))
            row["qt_gauss_std_ms"] = float(np.std(arr))
        else:
            row["n_beats_gauss"] = 0
            row["qt_gauss_min_ms"] = np.nan
            row["qt_gauss_max_ms"] = np.nan
            row["qt_gauss_mean_ms"] = np.nan
            row["qt_gauss_median_ms"] = np.nan
            row["qt_gauss_std_ms"] = np.nan

        # Average of all methods stats
        if qt_avg_values:
            arr = np.array(qt_avg_values)
            row["n_beats_avg"] = len(arr)
            row["qt_avg_min_ms"] = float(np.min(arr))
            row["qt_avg_max_ms"] = float(np.max(arr))
            row["qt_avg_mean_ms"] = float(np.mean(arr))
            row["qt_avg_median_ms"] = float(np.median(arr))
            row["qt_avg_std_ms"] = float(np.std(arr))
        else:
            row["n_beats_avg"] = 0
            row["qt_avg_min_ms"] = np.nan
            row["qt_avg_max_ms"] = np.nan
            row["qt_avg_mean_ms"] = np.nan
            row["qt_avg_median_ms"] = np.nan
            row["qt_avg_std_ms"] = np.nan

        rows.append(row)

    logger.info(
        "QT interval estimation: %d channels with valid measurements",
        len(rows)
    )

    return pd.DataFrame(rows)


def plot_qt_detection(
    waveform: np.ndarray,
    sample_rate: float,
    *,
    t_search_start_ms: float = 50.0,
    t_search_end_ms: float = 500.0,
    derivative_threshold: float = 0.1,
    baseline_threshold: float = 0.1,
    smoothing_ms: float = 5.0,
    figsize: Tuple[float, float] = (14, 6),
    show: bool = True,
) -> Tuple["Figure", "Axes"]:
    """Plot QT interval detection for a single waveform.

    Visualizes the waveform with detected Q point, T-end from all three methods,
    search region, tangent line, and computed QT values.

    Parameters
    ----------
    waveform : np.ndarray
        1-D waveform snippet centered on the depolarization peak.
    sample_rate : float
        Sampling rate in Hz.
    t_search_start_ms : float, optional
        Start searching for T-end this many ms after Q point. Default 50 ms.
    t_search_end_ms : float, optional
        Stop searching for T-end this many ms after Q point. Default 500 ms.
    derivative_threshold : float, optional
        Fraction of max derivative magnitude for T-end detection. Default 0.1.
    baseline_threshold : float, optional
        Fraction of T-wave amplitude for baseline crossing detection. Default 0.1.
    smoothing_ms : float, optional
        Smoothing window for derivative in ms. Default 5 ms.
    figsize : (float, float), optional
        Figure size in inches.
    show : bool, optional
        If ``True``, call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    # Run QT detection
    result = estimate_qt_interval(
        waveform,
        sample_rate,
        t_search_start_ms=t_search_start_ms,
        t_search_end_ms=t_search_end_ms,
        derivative_threshold=derivative_threshold,
        baseline_threshold=baseline_threshold,
        smoothing_ms=smoothing_ms,
    )

    q_idx = result["q_idx"]
    t_peak_idx = result["t_peak_idx"]
    t_end_deriv_idx = result["t_end_deriv_idx"]
    t_end_baseline_idx = result["t_end_baseline_idx"]
    t_end_tangent_idx = result["t_end_tangent_idx"]
    t_end_gauss_idx = result["t_end_gauss_idx"]
    qt_deriv_ms = result["qt_deriv_ms"]
    qt_baseline_ms = result["qt_baseline_ms"]
    qt_tangent_ms = result["qt_tangent_ms"]
    qt_gauss_ms = result["qt_gauss_ms"]
    qt_avg_ms = result["qt_avg_ms"]
    tangent_slope = result["tangent_slope"]
    tangent_intercept = result["tangent_intercept"]
    gauss_params = result["gauss_params"]

    samples_per_ms = sample_rate / 1000.0
    n_samples = len(waveform)

    # Compute search region indices
    t_start_idx = q_idx + int(t_search_start_ms * samples_per_ms)
    t_end_search_idx = min(n_samples, q_idx + int(t_search_end_ms * samples_per_ms))

    # Create time axis in ms
    time_ms = np.arange(n_samples) / samples_per_ms

    fig, ax = plt.subplots(figsize=figsize)

    # Plot waveform
    ax.plot(time_ms, waveform, "b-", linewidth=1.5, label="Waveform")

    # Highlight search region
    if t_start_idx < n_samples:
        ax.axvspan(
            time_ms[t_start_idx],
            time_ms[min(t_end_search_idx - 1, n_samples - 1)],
            alpha=0.12,
            color="yellow",
            label="T-end search region",
        )

    # Mark Q point
    ax.axvline(time_ms[q_idx], color="red", linestyle="--", linewidth=1.5, label="Q point")
    ax.plot(time_ms[q_idx], waveform[q_idx], "ro", markersize=10)

    # Mark T-wave peak
    if t_peak_idx >= 0 and t_peak_idx < n_samples:
        ax.plot(time_ms[t_peak_idx], waveform[t_peak_idx], "m^", markersize=8, label="T-peak")

    # Mark T-end from derivative method (green)
    if t_end_deriv_idx >= 0 and t_end_deriv_idx < n_samples:
        ax.axvline(time_ms[t_end_deriv_idx], color="green", linestyle="--", linewidth=1.5,
                   label=f"T-end (deriv): {qt_deriv_ms:.1f} ms")
        ax.plot(time_ms[t_end_deriv_idx], waveform[t_end_deriv_idx], "go", markersize=8)

    # Mark T-end from baseline method (orange)
    if t_end_baseline_idx >= 0 and t_end_baseline_idx < n_samples:
        ax.axvline(time_ms[t_end_baseline_idx], color="orange", linestyle=":", linewidth=2,
                   label=f"T-end (baseline): {qt_baseline_ms:.1f} ms")
        ax.plot(time_ms[t_end_baseline_idx], waveform[t_end_baseline_idx], "o",
                color="orange", markersize=8)

    # Mark T-end from tangent method (purple) and draw tangent line
    if t_end_tangent_idx >= 0 and t_end_tangent_idx < n_samples:
        ax.axvline(time_ms[t_end_tangent_idx], color="purple", linestyle="-.", linewidth=2,
                   label=f"T-end (tangent): {qt_tangent_ms:.1f} ms")
        ax.plot(time_ms[t_end_tangent_idx], waveform[t_end_tangent_idx], "s",
                color="purple", markersize=8)

        # Draw tangent line if we have valid slope/intercept
        if not np.isnan(tangent_slope) and not np.isnan(tangent_intercept):
            # Draw tangent line from T-peak to T-end (tangent)
            if t_peak_idx >= 0:
                x_start = t_peak_idx
                x_end = min(t_end_tangent_idx + int(20 * samples_per_ms), n_samples - 1)
                x_range = np.arange(x_start, x_end)
                y_tangent = tangent_slope * x_range + tangent_intercept
                ax.plot(time_ms[x_range], y_tangent, "purple", linestyle="--",
                        linewidth=1, alpha=0.7, label="Tangent line")

    # Mark T-end from Gaussian method (cyan) and draw Gaussian fit
    if t_end_gauss_idx >= 0 and t_end_gauss_idx < n_samples:
        ax.axvline(time_ms[t_end_gauss_idx], color="cyan", linestyle="-.", linewidth=2,
                   label=f"T-end (gauss): {qt_gauss_ms:.1f} ms")
        ax.plot(time_ms[t_end_gauss_idx], waveform[t_end_gauss_idx], "d",
                color="cyan", markersize=8)

        # Draw Gaussian fit curve if we have valid parameters
        if gauss_params is not None:
            amp = gauss_params["amplitude"]
            mu = gauss_params["mu"]
            sigma = gauss_params["sigma"]
            offset = gauss_params["offset"]

            # Draw Gaussian in the search region
            x_gauss = np.arange(t_end_search_idx - t_start_idx)
            y_gauss = amp * np.exp(-0.5 * ((x_gauss - mu) / sigma) ** 2) + offset
            ax.plot(time_ms[t_start_idx:t_end_search_idx], y_gauss, "cyan",
                    linestyle="--", linewidth=1.5, alpha=0.7, label="Gaussian fit")

    # Build title
    title_parts = []
    if not np.isnan(qt_deriv_ms):
        title_parts.append(f"D:{qt_deriv_ms:.0f}")
    if not np.isnan(qt_baseline_ms):
        title_parts.append(f"B:{qt_baseline_ms:.0f}")
    if not np.isnan(qt_tangent_ms):
        title_parts.append(f"T:{qt_tangent_ms:.0f}")
    if not np.isnan(qt_gauss_ms):
        title_parts.append(f"G:{qt_gauss_ms:.0f}")
    if not np.isnan(qt_avg_ms):
        title_parts.append(f"Avg:{qt_avg_ms:.0f}")

    if title_parts:
        title = "QT (ms) — " + " | ".join(title_parts)
    else:
        title = "QT Interval: Detection failed"

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Amplitude (µV)", fontsize=10)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Add text annotation with details
    info_lines = [f"Q idx: {q_idx}"]
    if t_end_deriv_idx >= 0:
        info_lines.append(f"Deriv: {qt_deriv_ms:.1f} ms")
    if t_end_baseline_idx >= 0:
        info_lines.append(f"Baseline: {qt_baseline_ms:.1f} ms")
    if t_end_tangent_idx >= 0:
        info_lines.append(f"Tangent: {qt_tangent_ms:.1f} ms")
    if t_end_gauss_idx >= 0:
        info_lines.append(f"Gaussian: {qt_gauss_ms:.1f} ms")
    if not np.isnan(qt_avg_ms):
        info_lines.append(f"Average: {qt_avg_ms:.1f} ms")

    info_text = "\n".join(info_lines)
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax
