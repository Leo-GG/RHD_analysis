"""
Quick visualization helpers for multi-channel electrophysiology data.

These functions produce :mod:`matplotlib` figures suitable for rapid
quality-checking of recordings. They return ``(fig, axes)`` tuples so
you can further customise the plots or embed them in a GUI.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


def _require_matplotlib() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with:  pip install matplotlib"
        )


def plot_channels(
    amplifier_data: np.ndarray,
    *,
    sample_rate: float = 20_000.0,
    start_seconds: float = 2.0,
    duration_seconds: float = 20.0,
    y_min: float = -400.0,
    y_max: float = 100.0,
    grid_shape: Tuple[int, int] = (8, 8),
    figsize: Tuple[float, float] = (18, 12),
    show: bool = True,
) -> Tuple["Figure", np.ndarray]:
    """Plot an overview grid of all channels.

    Displays a time window of each channel in a subplot grid, useful for
    quickly checking signal quality across a multi-electrode array.

    Parameters
    ----------
    amplifier_data : np.ndarray
        Shape ``(n_channels, n_samples)`` — amplifier data in µV.
    sample_rate : float, optional
        Sampling rate in Hz. Default 20 000.
    start_seconds : float, optional
        Start of the display window in seconds. Default 2 (skip transients).
    duration_seconds : float, optional
        Length of the display window in seconds. Default 20.
    y_min, y_max : float, optional
        Y-axis limits in µV. Defaults −400 / +100.
    grid_shape : (int, int), optional
        ``(rows, cols)`` of the subplot grid. Default ``(8, 8)`` for a 64-ch
        MEA.
    figsize : (float, float), optional
        Figure size in inches.
    show : bool, optional
        If ``True`` (default), call ``plt.show()`` before returning.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    _require_matplotlib()

    n_rows, n_cols = grid_shape
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=figsize,
        sharey="row",
        sharex="col",
    )
    ax_flat = axes.ravel()

    ti = int(start_seconds * sample_rate)
    te = ti + int(duration_seconds * sample_rate)
    n_channels = min(amplifier_data.shape[0], n_rows * n_cols)

    time_axis = np.arange(te - ti) / sample_rate + start_seconds

    for ch in range(n_channels):
        ax = ax_flat[ch]
        ax.plot(time_axis, amplifier_data[ch, ti:te], linewidth=0.4)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Ch {ch}", fontsize=7, pad=2)
        ax.tick_params(labelsize=5)

    # Hide unused axes
    for idx in range(n_channels, len(ax_flat)):
        ax_flat[idx].set_visible(False)

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes


def plot_waveforms(
    waveforms: Dict[int, List[np.ndarray]],
    average_waveforms: Optional[Dict[int, np.ndarray]] = None,
    *,
    channels: Optional[List[int]] = None,
    exclude_channels: Optional[List[int]] = None,
    sample_rate: float = 20_000.0,
    figsize: Tuple[float, float] = (14, 10),
    show: bool = True,
) -> Tuple["Figure", np.ndarray]:
    """Plot detected spike waveforms for selected channels.

    Parameters
    ----------
    waveforms : dict[int, list of np.ndarray]
        Per-channel waveform snippets (from :func:`~intan_reader.processing.spikes.extract_waveforms`).
    average_waveforms : dict[int, np.ndarray], optional
        Per-channel mean waveform. If provided, overlaid in red.
    channels : list of int, optional
        Which channels to plot. If ``None``, plots all channels that have
        at least one waveform.
    exclude_channels : list of int, optional
        Channels to exclude from plotting (e.g., noisy channels).
    sample_rate : float, optional
        Sampling rate in Hz (used only for the time axis label).
    figsize : (float, float), optional
        Figure size in inches.
    show : bool, optional
        If ``True``, call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    _require_matplotlib()

    if channels is None:
        channels = sorted(ch for ch, w in waveforms.items() if len(w) > 0)

    if exclude_channels is not None:
        exclude_set = set(exclude_channels)
        channels = [ch for ch in channels if ch not in exclude_set]

    n = len(channels)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No waveforms to display", ha="center", va="center")
        if show:
            plt.show()
        return fig, np.array([ax])

    n_cols = min(n, 8)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    ax_flat = axes.ravel()

    for idx, ch in enumerate(channels):
        ax = ax_flat[idx]
        for w in waveforms[ch]:
            t_ms = np.arange(len(w)) / sample_rate * 1000
            ax.plot(t_ms, w, color="0.7", linewidth=0.3)
        if average_waveforms and ch in average_waveforms:
            t_ms = np.arange(len(average_waveforms[ch])) / sample_rate * 1000
            ax.plot(t_ms, average_waveforms[ch], color="red", linewidth=1.2)
        ax.set_title(f"Ch {ch} ({len(waveforms[ch])} spikes)", fontsize=8)
        ax.set_xlabel("ms", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(len(channels), len(ax_flat)):
        ax_flat[idx].set_visible(False)

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes
