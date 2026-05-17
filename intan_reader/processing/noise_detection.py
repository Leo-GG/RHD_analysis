"""
Noise channel detection for multi-electrode array recordings.

Provides methods to identify channels that contain only noise (no neural signal)
using the dip test — channels with unimodal amplitude distributions are likely noise-only.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import diptest

logger = logging.getLogger(__name__)


def compute_dip_statistic(
    amplifier_data: np.ndarray,
    *,
    subsample: int = 10000,
) -> np.ndarray:
    """Compute dip statistic for each channel's amplitude distribution.

    Channels with low dip values have unimodal (Gaussian-like) distributions,
    suggesting they contain only noise. Channels with neural signals typically
    have multimodal distributions (baseline + spikes).

    Parameters
    ----------
    amplifier_data : np.ndarray
        Shape ``(n_channels, n_samples)`` — amplifier data in µV.
    subsample : int, optional
        Number of samples to use per channel (for speed). Default 10000.

    Returns
    -------
    np.ndarray
        Dip statistic for each channel. Shape ``(n_channels,)``.
        Low values (~0.01-0.03) suggest noise-only; higher values suggest signal.
    """
    n_channels, n_samples = amplifier_data.shape
    dip_values = np.zeros(n_channels)
    dip_p_values = np.zeros(n_channels)

    for ch in range(n_channels):
        data = amplifier_data[ch, :]
        dip_stat, dip_p_value = diptest.diptest(data)
        dip_values[ch] = dip_stat
        dip_p_values[ch] = dip_p_value
    return dip_values, dip_p_values


def detect_noisy_channels(
    amplifier_data: np.ndarray,
    *,
    dip_threshold: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Detect channels that likely contain only noise (no neural signal).

    Uses the dip test: channels with unimodal amplitude distributions
    (high p-value, failing to reject unimodality) are likely noise-only.
    Neural signals produce multimodal distributions (baseline + spikes).

    Parameters
    ----------
    amplifier_data : np.ndarray
        Shape ``(n_channels, n_samples)`` — amplifier data in µV.
    dip_threshold : float, optional
        Channels with dip test p-value above this are flagged as noisy.
        Default 0.05.

    Returns
    -------
    dict
        Dictionary with keys:

        - **is_noisy** : bool array, True for noisy channels
        - **dip_values** : dip statistic per channel
        - **dip_p_values** : dip test p-value per channel
    """
    n_channels = amplifier_data.shape[0]

    logger.info("Computing dip statistics for noise detection...")
    dip_values, dip_p_values = compute_dip_statistic(amplifier_data)
    is_noisy = dip_p_values > dip_threshold

    n_flagged = is_noisy.sum()
    logger.info(
        "Dip test: %d/%d channels flagged as noisy (p > %.3f)",
        n_flagged, n_channels, dip_threshold
    )

    return {
        "is_noisy": is_noisy,
        "dip_values": dip_values,
        "dip_p_values": dip_p_values,
    }
