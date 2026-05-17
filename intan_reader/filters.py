"""
Signal filtering utilities for electrophysiology data.

All functions accept either a 1-D signal or a 2-D array of shape
``(n_channels, n_samples)`` and return an array of the same shape.

Filters are implemented with :mod:`scipy.signal` using zero-phase
forward-backward filtering (:func:`~scipy.signal.filtfilt`) so that
the output has no group delay.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy import signal as _sig


def notch(
    data: np.ndarray,
    sample_rate: float,
    freq: float,
    quality: float = 30.0,
) -> np.ndarray:
    """Apply a zero-phase notch (band-stop) filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal — shape ``(n_samples,)`` or ``(n_channels, n_samples)``.
    sample_rate : float
        Sampling rate in Hz.
    freq : float
        Centre frequency to remove, in Hz (e.g. 50 or 60).
    quality : float, optional
        Quality factor *Q* of the notch filter. Higher values give a narrower
        notch. Default is 30.

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as *data*.
    """
    b, a = _sig.iirnotch(w0=freq, Q=quality, fs=sample_rate)
    return _sig.filtfilt(b, a, data).astype(data.dtype)


def highpass(
    data: np.ndarray,
    sample_rate: float,
    cutoff: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal — shape ``(n_samples,)`` or ``(n_channels, n_samples)``.
    sample_rate : float
        Sampling rate in Hz.
    cutoff : float
        High-pass cutoff frequency in Hz.
    order : int, optional
        Filter order. Default is 4. Higher orders give sharper roll-off but
        may introduce numerical instability.

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as *data*.
    """
    b, a = _sig.butter(N=order, Wn=cutoff, fs=sample_rate, btype="high")
    return _sig.filtfilt(b, a, data).astype(data.dtype)


def lowpass(
    data: np.ndarray,
    sample_rate: float,
    cutoff: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal — shape ``(n_samples,)`` or ``(n_channels, n_samples)``.
    sample_rate : float
        Sampling rate in Hz.
    cutoff : float
        Low-pass cutoff frequency in Hz.
    order : int, optional
        Filter order. Default is 4.

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as *data*.
    """
    b, a = _sig.butter(N=order, Wn=cutoff, fs=sample_rate, btype="low")
    return _sig.filtfilt(b, a, data).astype(data.dtype)


def bandpass(
    data: np.ndarray,
    sample_rate: float,
    low_cutoff: float,
    high_cutoff: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal — shape ``(n_samples,)`` or ``(n_channels, n_samples)``.
    sample_rate : float
        Sampling rate in Hz.
    low_cutoff : float
        Lower edge of the passband in Hz.
    high_cutoff : float
        Upper edge of the passband in Hz.
    order : int, optional
        Filter order. Default is 4.

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as *data*.
    """
    b, a = _sig.butter(
        N=order, Wn=[low_cutoff, high_cutoff], fs=sample_rate, btype="band"
    )
    return _sig.filtfilt(b, a, data).astype(data.dtype)
