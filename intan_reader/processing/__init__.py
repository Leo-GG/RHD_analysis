"""
intan_reader.processing - Signal processing for electrophysiology data.

Provides artifact detection/rejection and spike detection/filtering routines.
"""

from intan_reader.processing.artifacts import detect_artifacts, detect_artifacts_threshold
from intan_reader.processing.spikes import detect_peaks, filter_peaks, extract_waveforms, compute_spike_statistics
from intan_reader.processing.noise_detection import (
    detect_noisy_channels,
    compute_dip_statistic,
)
from intan_reader.processing.cardiac import (
    estimate_qt_interval,
    compute_qt_intervals,
    plot_qt_detection,
)

__all__ = [
    "detect_artifacts",
    "detect_artifacts_threshold",
    "detect_peaks",
    "filter_peaks",
    "extract_waveforms",
    "compute_spike_statistics",
    "detect_noisy_channels",
    "compute_dip_statistic",
    "estimate_qt_interval",
    "compute_qt_intervals",
    "plot_qt_detection",
]
