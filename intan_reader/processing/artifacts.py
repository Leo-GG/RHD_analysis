"""
Artifact detection for multi-channel electrophysiology data.

Provides two strategies:

1. **Envelope-based** (:func:`detect_artifacts`) — smooths the rectified
   signal with a moving average and flags regions whose envelope exceeds
   a multiple of the standard deviation.
2. **Threshold-based** (:func:`detect_artifacts_threshold`) — flags any
   sample whose absolute amplitude exceeds a fixed threshold.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


def detect_artifacts(
    amplifier_data: np.ndarray,
    *,
    window_samples: int = 20_000,
    threshold_uv: float = 300.0,
) -> List[np.ndarray]:
    """Detect artifacts using a smoothed-envelope method.

    For each channel the rectified signal is convolved with a uniform
    kernel of length *window_samples*. Any sample where the smoothed
    envelope exceeds ``n_std × std(envelope)`` is marked as an artifact.

    Parameters
    ----------
    amplifier_data : np.ndarray
        Shape ``(n_channels, n_samples)`` — amplifier data in µV.
    window_samples : int, optional
        Length of the smoothing window in samples. At 20 kS/s the default
        of 20 000 corresponds to 1 second.
    n_std : float, optional
        Number of standard deviations above the mean envelope to set the
        artifact threshold. Default is 3.

    Returns
    -------
    list of np.ndarray
        One boolean array per channel (``True`` = artifact).
    """
    logger.info(
        "Running envelope artifact detection (window=%d, n_std=%.1f)",
        window_samples,
        threshold_uv,
    )
    n_channels = amplifier_data.shape[0]
    kernel = np.ones(window_samples) / window_samples
    artifacts: List[np.ndarray] = []

    for ch in range(n_channels):
        envelope = uniform_filter1d(
            np.abs(amplifier_data[ch, :]), window_samples, mode="reflect"
        )
        #threshold = np.std(envelope) * n_std
        artifacts.append(envelope > threshold_uv)

    return artifacts


def detect_artifacts_threshold(
    amplifier_data: np.ndarray,
    *,
    threshold_uv: float = 300.0,
) -> List[np.ndarray]:
    """Flag samples whose absolute amplitude exceeds a fixed threshold.

    Parameters
    ----------
    amplifier_data : np.ndarray
        Shape ``(n_channels, n_samples)`` — amplifier data in µV.
    threshold_uv : float, optional
        Absolute amplitude threshold in µV. Default is 300.

    Returns
    -------
    list of np.ndarray
        One boolean array per channel (``True`` = artifact).
    """
    logger.info("Running threshold artifact detection (threshold=%.1f µV)", threshold_uv)
    n_channels = amplifier_data.shape[0]
    return [np.abs(amplifier_data[ch, :]) > threshold_uv for ch in range(n_channels)]


def no_artifacts(amplifier_data: np.ndarray) -> List[np.ndarray]:
    """Return an all-clean artifact mask (no artifacts flagged).

    Useful as a pass-through when artifact rejection is not needed.

    Parameters
    ----------
    amplifier_data : np.ndarray
        Shape ``(n_channels, n_samples)``.

    Returns
    -------
    list of np.ndarray
        One all-``False`` boolean array per channel.
    """
    n_channels = amplifier_data.shape[0]
    n_samples = amplifier_data.shape[1]
    return [np.zeros(n_samples, dtype=bool) for _ in range(n_channels)]
