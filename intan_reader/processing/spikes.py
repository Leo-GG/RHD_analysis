"""
Spike detection, filtering, and waveform extraction.

The typical workflow is:

1. :func:`detect_peaks` — find candidate negative-going peaks on each channel.
2. :func:`filter_peaks` — reject peaks that fall in artifact zones, are out
   of amplitude range, or have atypical waveform shapes.
3. :func:`extract_waveforms` — cut fixed-length snippets around each peak.

A convenience wrapper :func:`get_peaks` chains steps 1–2.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as _sig

logger = logging.getLogger(__name__)


def detect_peaks(
    amplifier_data: np.ndarray,
    artifacts: List[np.ndarray],
    *,
    threshold_std: float = 3.5,
    min_distance: int = 5000,
) -> Dict[int, np.ndarray]:
    """Detect negative-going peaks on each channel.

    A peak is accepted when its (inverted) amplitude exceeds
    ``threshold_std × std`` of the artifact-free portion of the channel.

    Parameters
    ----------
    amplifier_data : np.ndarray
        Shape ``(n_channels, n_samples)`` — amplifier data in µV.
    artifacts : list of np.ndarray
        Per-channel boolean artifact masks (from :mod:`~intan_reader.processing.artifacts`).
    threshold_std : float, optional
        Detection threshold in multiples of the channel standard deviation.
        Default is 3.5.
    min_distance : int, optional
        Minimum distance between two peaks in samples. Default is 5000
        (250 ms at 20 kS/s).

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from channel index to an array of sample indices where peaks
        were detected.
    """
    n_channels = amplifier_data.shape[0]
    peaks: Dict[int, np.ndarray] = {}

    for ch in range(n_channels):
        clean_mask = ~artifacts[ch].astype(bool)
        channel_std = np.std(amplifier_data[ch, clean_mask]) if clean_mask.any() else 1.0
        height = threshold_std * channel_std

        found, _ = _sig.find_peaks(
            -amplifier_data[ch, :],
            height=height,
            distance=min_distance,
        )
        peaks[ch] = found

    return peaks


def filter_peaks(
    amplifier_data: np.ndarray,
    peaks: Dict[int, np.ndarray],
    artifacts: List[np.ndarray],
    *,
    min_amplitude_uv: float = 50.0,
    max_amplitude_uv: float = 500.0,
    edge_margin: int = 10_000,
    waveform_half_width: int = 5000,
    max_z_score: float = 2.0,
) -> Dict[int, np.ndarray]:
    """Reject peaks by amplitude, artifact overlap, edge proximity, and waveform shape.

    Steps applied per channel:

    1. Discard peaks inside artifact regions or within *edge_margin* samples
       of the recording boundaries.
    2. Discard peaks whose amplitude is outside
       [*min_amplitude_uv*, *max_amplitude_uv*].
    3. Compute the RMSD of each waveform snippet relative to the channel
       average waveform; discard outliers with z-score > *max_z_score*.

    Parameters
    ----------
    amplifier_data : np.ndarray
        Shape ``(n_channels, n_samples)``.
    peaks : dict[int, np.ndarray]
        Raw peak indices per channel (from :func:`detect_peaks`).
    artifacts : list of np.ndarray
        Per-channel boolean artifact masks.
    min_amplitude_uv : float, optional
        Minimum peak amplitude (absolute value) in µV. Default 50.
    max_amplitude_uv : float, optional
        Maximum peak amplitude (absolute value) in µV. Default 500.
    edge_margin : int, optional
        Number of samples to exclude at each edge of the recording.
    waveform_half_width : int, optional
        Half-width (in samples) of the snippet cut around each peak for
        shape comparison. Default 5000.
    max_z_score : float, optional
        Maximum z-score of waveform RMSD to keep a peak. Default 2.0.

    Returns
    -------
    dict[int, np.ndarray]
        Filtered peak indices per channel.
    """
    n_channels = len(artifacts)
    n_samples = len(artifacts[0])
    filtered: Dict[int, np.ndarray] = {}

    for ch in range(n_channels):
        # Step 1 & 2: amplitude + location filtering
        candidates = []
        for p in peaks.get(ch, []):
            if artifacts[ch][p]:
                continue
            if p < edge_margin or p >= n_samples - edge_margin:
                continue
            amp = amplifier_data[ch, p]
            if amp < -max_amplitude_uv or amp > -min_amplitude_uv:
                continue
            candidates.append(p)

        # Step 3: waveform-shape outlier rejection
        if len(candidates) > 1:
            waves = np.array(
                [
                    amplifier_data[ch, p - waveform_half_width : p + waveform_half_width]
                    for p in candidates
                    if p - waveform_half_width >= 0
                    and p + waveform_half_width <= n_samples
                ]
            )
            if waves.ndim == 2 and waves.shape[0] > 1:
                mean_wave = waves.mean(axis=0)
                rmsd = np.sqrt(np.mean((waves - mean_wave) ** 2, axis=1))
                z_scores = (rmsd - rmsd.mean()) / (rmsd.std() + 1e-12)
                keep = z_scores < max_z_score
                candidates = [c for c, k in zip(candidates, keep) if k]

        filtered[ch] = np.array(candidates, dtype=int)

    return filtered


def get_peaks(
    amplifier_data: np.ndarray,
    artifacts: List[np.ndarray],
    *,
    threshold_std: float = 3.5,
    min_amplitude_uv: float = 50.0,
    max_amplitude_uv: float = 500.0,
    max_z_score: float = 2.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Convenience wrapper: detect then filter peaks.

    Parameters
    ----------
    amplifier_data : np.ndarray
        Shape ``(n_channels, n_samples)``.
    artifacts : list of np.ndarray
        Per-channel boolean artifact masks.
    threshold_std : float, optional
        Detection threshold (std multiples).
    min_amplitude_uv : float, optional
        Minimum peak amplitude in µV.
    max_amplitude_uv : float, optional
        Maximum peak amplitude in µV.
    max_z_score : float, optional
        Maximum waveform-RMSD z-score to keep a peak. Default 2.0.

    Returns
    -------
    raw_peaks : dict[int, np.ndarray]
        All detected peaks.
    filtered_peaks : dict[int, np.ndarray]
        Peaks after filtering.
    """
    raw = detect_peaks(amplifier_data, artifacts, threshold_std=threshold_std)
    filt = filter_peaks(
        amplifier_data,
        raw,
        artifacts,
        min_amplitude_uv=min_amplitude_uv,
        max_amplitude_uv=max_amplitude_uv,
        max_z_score=max_z_score,
    )
    return raw, filt


def compute_spike_statistics(
    waveforms: Dict[int, List[np.ndarray]],
    peaks: Dict[int, np.ndarray],
    sample_rate: float,
    *,
    exclude_channels: Optional[List[int]] = None,
) -> "pd.DataFrame":
    """Compute per-channel spike statistics.

    Parameters
    ----------
    waveforms : dict[int, list of np.ndarray]
        Per-channel waveform snippets (from :func:`extract_waveforms`).
    peaks : dict[int, np.ndarray]
        Peak sample indices per channel.
    sample_rate : float
        Sampling rate in Hz.
    exclude_channels : list of int, optional
        Channels to exclude (e.g., noisy channels).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        - **channel** : channel index
        - **n_spikes** : number of spikes
        - **freq_hz** : spike frequency (spikes per second)
        - **amp_min** : minimum peak amplitude (µV)
        - **amp_max** : maximum peak amplitude (µV)
        - **amp_mean** : mean peak amplitude (µV)
        - **amp_median** : median peak amplitude (µV)
        - **amp_std** : std of peak amplitudes (µV)
        - **isi_min_ms** : minimum inter-spike interval (ms)
        - **isi_max_ms** : maximum inter-spike interval (ms)
        - **isi_mean_ms** : mean inter-spike interval (ms)
        - **isi_median_ms** : median inter-spike interval (ms)
        - **isi_std_ms** : std of inter-spike intervals (ms)
        - **wf_dev_min** : minimum waveform deviation from average (RMSD, µV)
        - **wf_dev_max** : maximum waveform deviation from average (RMSD, µV)
        - **wf_dev_mean** : mean waveform deviation from average (RMSD, µV)
        - **wf_dev_median** : median waveform deviation from average (RMSD, µV)
    """
    import pandas as pd

    exclude_set = set(exclude_channels) if exclude_channels else set()

    rows = []
    for ch in sorted(waveforms.keys()):
        if ch in exclude_set:
            continue

        waves = waveforms[ch]
        ch_peaks = peaks.get(ch, np.array([]))
        n_spikes = len(waves)

        if n_spikes == 0:
            continue

        # Peak amplitudes (minimum of each waveform, since spikes are negative-going)
        amplitudes = np.array([np.min(w) for w in waves])

        # Compute recording duration from peak positions for frequency
        if len(ch_peaks) >= 2:
            duration_samples = ch_peaks[-1] - ch_peaks[0]
            duration_sec = duration_samples / sample_rate
            freq_hz = (n_spikes - 1) / duration_sec if duration_sec > 0 else 0.0
        else:
            freq_hz = 0.0

        # Inter-spike intervals
        if len(ch_peaks) >= 2:
            isi_samples = np.diff(np.sort(ch_peaks))
            isi_ms = isi_samples / sample_rate * 1000
            isi_min = float(np.min(isi_ms))
            isi_max = float(np.max(isi_ms))
            isi_mean = float(np.mean(isi_ms))
            isi_median = float(np.median(isi_ms))
            isi_std = float(np.std(isi_ms))
        else:
            isi_min = isi_max = isi_mean = isi_median = isi_std = np.nan

        # Waveform deviation from average (RMSD)
        if n_spikes >= 2:
            waves_arr = np.array(waves)
            avg_waveform = np.mean(waves_arr, axis=0)
            # RMSD for each waveform vs average
            deviations = np.sqrt(np.mean((waves_arr - avg_waveform) ** 2, axis=1))
            wf_dev_min = float(np.min(deviations))
            wf_dev_max = float(np.max(deviations))
            wf_dev_mean = float(np.mean(deviations))
            wf_dev_median = float(np.median(deviations))
        else:
            wf_dev_min = wf_dev_max = wf_dev_mean = wf_dev_median = np.nan

        rows.append({
            "channel": ch,
            "n_spikes": n_spikes,
            "freq_hz": freq_hz,
            "amp_min": float(np.min(amplitudes)),
            "amp_max": float(np.max(amplitudes)),
            "amp_mean": float(np.mean(amplitudes)),
            "amp_median": float(np.median(amplitudes)),
            "amp_std": float(np.std(amplitudes)),
            "isi_min_ms": isi_min,
            "isi_max_ms": isi_max,
            "isi_mean_ms": isi_mean,
            "isi_median_ms": isi_median,
            "isi_std_ms": isi_std,
            "wf_dev_min": wf_dev_min,
            "wf_dev_max": wf_dev_max,
            "wf_dev_mean": wf_dev_mean,
            "wf_dev_median": wf_dev_median,
        })

    return pd.DataFrame(rows)


def extract_waveforms(
    amplifier_data: np.ndarray,
    peaks: Dict[int, np.ndarray],
    *,
    half_width: int = 5000,
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, np.ndarray]]:
    """Cut fixed-length waveform snippets centred on each peak.

    Parameters
    ----------
    amplifier_data : np.ndarray
        Shape ``(n_channels, n_samples)`` or ``(n_samples,)`` for a single
        channel. If 1-D, *peaks* should map channel ``0`` to a peak array,
        or be a plain array.
    peaks : dict[int, np.ndarray] or np.ndarray
        Peak sample indices. If *amplifier_data* is 1-D, *peaks* may be a
        flat array of indices (treated as channel 0).
    half_width : int, optional
        Number of samples to extract on each side of the peak. Default 5000.

    Returns
    -------
    waveforms : dict[int, list of np.ndarray]
        Per-channel list of 1-D waveform arrays (length ``2 * half_width``).
    average_waveforms : dict[int, np.ndarray]
        Per-channel mean waveform. Only present for channels with ≥1 waveform.
    """
    waveforms: Dict[int, List[np.ndarray]] = {}
    averages: Dict[int, np.ndarray] = {}

    is_single = amplifier_data.ndim == 1
    if is_single:
        # Normalise to multi-channel interface
        amplifier_data = amplifier_data[np.newaxis, :]
        if not isinstance(peaks, dict):
            peaks = {0: np.asarray(peaks)}

    n_channels = amplifier_data.shape[0]
    n_samples = amplifier_data.shape[1]

    for ch in range(n_channels):
        ch_waves: List[np.ndarray] = []
        for p in peaks.get(ch, []):
            start = p - half_width
            end = p + half_width
            if start < 0 or end > n_samples:
                continue
            ch_waves.append(amplifier_data[ch, start:end])

        waveforms[ch] = ch_waves
        if ch_waves:
            averages[ch] = np.mean(ch_waves, axis=0)

    return waveforms, averages
