"""
High-level :class:`Recording` object — the primary public API.

A ``Recording`` wraps the raw data dictionary returned by
:func:`~intan_reader.io.read_rhd_file` and exposes convenient methods for
filtering, artifact detection, spike detection, and visualisation.

It is designed to be the single entry-point for interactive use, scripts,
and as a back-end for GUIs or web interfaces.

Examples
--------
Load a single file::

    from intan_reader import Recording

    rec = Recording.from_file("data/experiment_001.rhd")
    print(rec.num_channels, rec.duration_seconds)

Load and merge multiple files that share a base name::

    rec = Recording.from_folder("recordings/", "Exp1_condition_3")
    rec.detect_artifacts()
    raw_peaks, filtered_peaks = rec.detect_spikes(threshold=3.5)
    rec.plot_channels()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from intan_reader.io.rhd_reader import read_rhd_file
from intan_reader.processing.artifacts import (
    detect_artifacts as _detect_artifacts,
    detect_artifacts_threshold as _detect_artifacts_threshold,
    no_artifacts as _no_artifacts,
)
from intan_reader.processing.spikes import (
    detect_peaks as _detect_peaks,
    filter_peaks as _filter_peaks,
    extract_waveforms as _extract_waveforms,
    get_peaks as _get_peaks,
    compute_spike_statistics as _compute_spike_statistics,
)
from intan_reader.processing.noise_detection import (
    detect_noisy_channels as _detect_noisy_channels,
)
from intan_reader.processing.cardiac import (
    compute_qt_intervals as _compute_qt_intervals,
)

logger = logging.getLogger(__name__)


class Recording:
    """A loaded Intan RHD recording with processing utilities.

    You will typically create instances via the class methods
    :meth:`from_file` or :meth:`from_folder` rather than calling the
    constructor directly.

    Parameters
    ----------
    amplifier_data : np.ndarray
        Shape ``(n_channels, n_samples)`` — amplifier data in µV.
    sample_rate : float
        Sampling rate in Hz.
    metadata : dict, optional
        Any extra metadata from the RHD file (header, channel info, etc.).
    """

    def __init__(
        self,
        amplifier_data: np.ndarray,
        sample_rate: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.amplifier_data: np.ndarray = amplifier_data
        """Amplifier data array — shape ``(n_channels, n_samples)``, units µV."""

        self.sample_rate: float = sample_rate
        """Amplifier sampling rate in Hz."""

        self.metadata: Dict[str, Any] = metadata or {}
        """Additional metadata from the RHD file (header, frequency params, etc.)."""

        self._artifacts: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_channels(self) -> int:
        """Number of amplifier channels."""
        return self.amplifier_data.shape[0]

    @property
    def num_samples(self) -> int:
        """Total number of samples per channel."""
        return self.amplifier_data.shape[1]

    @property
    def duration_seconds(self) -> float:
        """Recording duration in seconds."""
        return self.num_samples / self.sample_rate

    @property
    def time_vector(self) -> np.ndarray:
        """Time axis in seconds, shape ``(n_samples,)``."""
        return np.arange(self.num_samples) / self.sample_rate

    @property
    def artifacts(self) -> List[np.ndarray]:
        """Per-channel artifact masks. Call :meth:`detect_artifacts` first.

        Returns a list of boolean arrays (one per channel). Accessing this
        property before running artifact detection returns all-``False``
        masks (i.e. no artifacts).
        """
        if self._artifacts is None:
            return _no_artifacts(self.amplifier_data)
        return self._artifacts

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        *,
        apply_notch: bool = True,
        notch_freq: Optional[float] = None,
        highpass_cutoff: Optional[float] = None,
        lowpass_cutoff: Optional[float] = None,
        sample_rate_override: Optional[float] = None,
    ) -> "Recording":
        """Load a single RHD file.

        Parameters
        ----------
        filepath : str or Path
            Path to the ``.rhd`` file.
        apply_notch : bool, optional
            Apply the notch filter that was active during acquisition.
            Ignored when *notch_freq* is explicitly set.
        notch_freq : float or None, optional
            Notch filter frequency in Hz (e.g. 50 or 60). Overrides the
            file header setting and *apply_notch*. Pass ``0`` to disable.
        highpass_cutoff : float or None, optional
            High-pass filter cutoff in Hz. Applied after loading.
        lowpass_cutoff : float or None, optional
            Low-pass filter cutoff in Hz. Applied after loading.
        sample_rate_override : float or None, optional
            Override the sample rate from the file header (Hz). Use when
            the recording was made at a rate different from what the
            header reports (e.g. 10 000 vs 20 000).

        Returns
        -------
        Recording
        """
        result = read_rhd_file(
            filepath,
            apply_notch=apply_notch,
            notch_freq=notch_freq,
            highpass_cutoff=highpass_cutoff,
            lowpass_cutoff=lowpass_cutoff,
            sample_rate_override=sample_rate_override,
        )
        return cls(
            amplifier_data=result["amplifier_data"],
            sample_rate=result["sample_rate"],
            metadata=result,
        )

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        base_name: str,
        *,
        max_files: Optional[int] = 3,
        apply_notch: bool = True,
        notch_freq: Optional[float] = None,
        highpass_cutoff: Optional[float] = None,
        lowpass_cutoff: Optional[float] = None,
        sample_rate_override: Optional[float] = None,
    ) -> "Recording":
        """Load and merge multiple RHD files from a folder.

        Files whose name contains *base_name* are sorted alphabetically
        and concatenated along the time axis. This reproduces the
        behaviour of the legacy ``process_intan.merge()`` function.

        Parameters
        ----------
        folder : str or Path
            Directory containing the ``.rhd`` files.
        base_name : str
            Substring that must appear in each filename to be included.
        max_files : int or None, optional
            Maximum number of files to load (default 3). Pass ``None`` to
            load all matching files.
        apply_notch : bool, optional
            Apply the notch filter during file reading. Ignored when
            *notch_freq* is explicitly set.
        notch_freq : float or None, optional
            Notch filter frequency in Hz. Overrides file header + *apply_notch*.
        highpass_cutoff : float or None, optional
            High-pass filter cutoff in Hz. Applied per file before merging.
        lowpass_cutoff : float or None, optional
            Low-pass filter cutoff in Hz. Applied per file before merging.
        sample_rate_override : float or None, optional
            Override the sample rate from the file header (Hz).

        Returns
        -------
        Recording

        Raises
        ------
        FileNotFoundError
            If no matching files are found in *folder*.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Directory not found: {folder}")

        matches = sorted(p for p in folder.iterdir() if base_name in p.name)
        if not matches:
            raise FileNotFoundError(
                f"No files matching '{base_name}' found in {folder}"
            )

        if max_files is not None:
            matches = matches[:max_files]

        logger.info("Merging %d file(s) from %s", len(matches), folder)

        merged_data: Optional[np.ndarray] = None
        last_result: Dict[str, Any] = {}

        for path in matches:
            logger.info("  Reading %s", path.name)
            result = read_rhd_file(
                path,
                apply_notch=apply_notch,
                notch_freq=notch_freq,
                highpass_cutoff=highpass_cutoff,
                lowpass_cutoff=lowpass_cutoff,
                sample_rate_override=sample_rate_override,
            )
            file_data = result["amplifier_data"]

            if merged_data is not None and merged_data.shape[0] == file_data.shape[0]:
                merged_data = np.concatenate((merged_data, file_data), axis=1)
            else:
                merged_data = file_data

            last_result = result

        last_result["amplifier_data"] = merged_data
        return cls(
            amplifier_data=merged_data,
            sample_rate=last_result["sample_rate"],
            metadata=last_result,
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def apply_notch(self, freq: float = 50.0, quality: float = 30.0) -> None:
        """Apply an in-place notch filter to the amplifier data.

        Parameters
        ----------
        freq : float
            Centre frequency in Hz (e.g. 50 or 60).
        quality : float
            Quality factor Q.
        """
        from intan_reader.filters import notch

        self.amplifier_data = notch(
            self.amplifier_data, self.sample_rate, freq, quality
        )

    def apply_highpass(self, cutoff: float, order: int = 4) -> None:
        """Apply an in-place high-pass filter.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency in Hz.
        order : int
            Butterworth filter order.
        """
        from intan_reader.filters import highpass

        self.amplifier_data = highpass(
            self.amplifier_data, self.sample_rate, cutoff, order
        )

    def apply_lowpass(self, cutoff: float, order: int = 4) -> None:
        """Apply an in-place low-pass filter.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency in Hz.
        order : int
            Butterworth filter order.
        """
        from intan_reader.filters import lowpass

        self.amplifier_data = lowpass(
            self.amplifier_data, self.sample_rate, cutoff, order
        )

    def apply_bandpass(
        self, low: float, high: float, order: int = 4
    ) -> None:
        """Apply an in-place band-pass filter.

        Parameters
        ----------
        low : float
            Lower cutoff in Hz.
        high : float
            Upper cutoff in Hz.
        order : int
            Butterworth filter order.
        """
        from intan_reader.filters import bandpass

        self.amplifier_data = bandpass(
            self.amplifier_data, self.sample_rate, low, high, order
        )

    # ------------------------------------------------------------------
    # Artifact detection
    # ------------------------------------------------------------------

    def detect_artifacts(
        self,
        *,
        method: str = "envelope",
        window_samples: int = 20_000,
        n_std: float = 3.0,
        threshold_uv: float = 300.0,
    ) -> List[np.ndarray]:
        """Detect artifacts and store the result on this recording.

        Parameters
        ----------
        method : ``"envelope"`` or ``"threshold"``
            Detection strategy. ``"envelope"`` (default) uses a smoothed
            rectified-signal envelope. ``"threshold"`` flags any sample
            exceeding a fixed amplitude.
        window_samples : int
            Smoothing window for the envelope method.
        n_std : float
            Number of std deviations for the envelope method.
        threshold_uv : float
            Amplitude threshold in µV for the threshold method.

        Returns
        -------
        list of np.ndarray
            Per-channel boolean masks (``True`` = artifact). Also stored in
            :attr:`artifacts`.
        """
        if method == "envelope":
            self._artifacts = _detect_artifacts(
                self.amplifier_data,
                window_samples=window_samples,
                threshold_uv=threshold_uv,
            )
        elif method == "threshold":
            self._artifacts = _detect_artifacts_threshold(
                self.amplifier_data,
                threshold_uv=threshold_uv,
            )
        else:
            raise ValueError(f"Unknown artifact method: {method!r}")

        return self._artifacts

    # ------------------------------------------------------------------
    # Noise channel detection
    # ------------------------------------------------------------------

    def detect_noisy_channels(
        self,
        *,
        dip_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Detect channels that likely contain only noise (no neural signal).

        Uses the dip test: channels with unimodal amplitude distributions
        (high p-value, failing to reject unimodality) are likely noise-only.
        Neural signals produce multimodal distributions (baseline + spikes).

        Parameters
        ----------
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
        return _detect_noisy_channels(
            self.amplifier_data,
            dip_threshold=dip_threshold,
        )

    # ------------------------------------------------------------------
    # Spike detection
    # ------------------------------------------------------------------

    def detect_spikes(
        self,
        *,
        threshold: float = 3.5,
        min_amplitude_uv: float = 50.0,
        max_amplitude_uv: float = 500.0,
        max_z_score: float = 2.0,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Detect and filter spikes on all channels.

        Runs :func:`~intan_reader.processing.spikes.detect_peaks` followed
        by :func:`~intan_reader.processing.spikes.filter_peaks`.

        Parameters
        ----------
        threshold : float
            Detection threshold in multiples of channel std.
        min_amplitude_uv : float
            Minimum peak amplitude in µV to keep.
        max_amplitude_uv : float
            Maximum peak amplitude in µV to keep.
        max_z_score : float
            Maximum waveform-RMSD z-score for shape-based outlier
            rejection. Peaks whose waveform deviates from the channel
            mean by more than this many standard deviations are
            discarded. Default 2.0.

        Returns
        -------
        raw_peaks : dict[int, np.ndarray]
            All detected peaks per channel.
        filtered_peaks : dict[int, np.ndarray]
            Peaks surviving amplitude + shape filtering.
        """
        return _get_peaks(
            self.amplifier_data,
            self.artifacts,
            threshold_std=threshold,
            min_amplitude_uv=min_amplitude_uv,
            max_amplitude_uv=max_amplitude_uv,
            max_z_score=max_z_score,
        )

    def extract_waveforms(
        self,
        peaks: Dict[int, np.ndarray],
        *,
        half_width: int = 5000,
        exclude_channels: Optional[List[int]] = None,
    ) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, np.ndarray]]:
        """Extract waveform snippets around detected peaks.

        Parameters
        ----------
        peaks : dict[int, np.ndarray]
            Peak sample indices per channel.
        half_width : int
            Number of samples on each side of the peak.
        exclude_channels : list of int, optional
            Channels to exclude (e.g., noisy channels from
            :meth:`detect_noisy_channels`).

        Returns
        -------
        waveforms : dict[int, list of np.ndarray]
        average_waveforms : dict[int, np.ndarray]
        """
        if exclude_channels is not None:
            exclude_set = set(exclude_channels)
            peaks = {ch: p for ch, p in peaks.items() if ch not in exclude_set}

        return _extract_waveforms(
            self.amplifier_data, peaks, half_width=half_width
        )

    def compute_spike_statistics(
        self,
        peaks: Dict[int, np.ndarray],
        *,
        half_width: int = 5000,
        exclude_channels: Optional[List[int]] = None,
    ) -> Any:
        """Compute per-channel spike statistics.

        Extracts waveforms and computes amplitude and timing metrics.

        Parameters
        ----------
        peaks : dict[int, np.ndarray]
            Peak indices (e.g. from :meth:`detect_spikes`).
        half_width : int
            Waveform half-width in samples for amplitude extraction.
        exclude_channels : list of int, optional
            Channels to exclude (e.g., noisy channels from
            :meth:`detect_noisy_channels`).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:

            - **channel** : channel index
            - **n_spikes** : number of spikes
            - **freq_hz** : spike frequency (spikes per second)
            - **amp_min/max/mean/median/std** : peak amplitude stats (µV)
            - **isi_min/max/mean/median/std_ms** : inter-spike interval stats (ms)
        """
        waves, _ = self.extract_waveforms(
            peaks, half_width=half_width, exclude_channels=exclude_channels
        )
        return _compute_spike_statistics(
            waves,
            peaks,
            self.sample_rate,
            exclude_channels=exclude_channels,
        )

    def compute_qt_intervals(
        self,
        peaks: Dict[int, np.ndarray],
        *,
        half_width: int = 5000,
        exclude_channels: Optional[List[int]] = None,
        t_search_start_ms: float = 50.0,
        t_search_end_ms: float = 500.0,
        derivative_threshold: float = 0.1,
        baseline_threshold: float = 0.1,
        smoothing_ms: float = 5.0,
    ) -> Any:
        """Compute QT intervals (field potential duration) for cardiac MEA data.

        Uses four methods to detect T-end after each depolarization peak:
        1. Derivative threshold method
        2. Baseline crossing method
        3. Tangent method (steepest descent intersects baseline)
        4. Gaussian fit method (T-end = mu + 2*sigma)

        Parameters
        ----------
        peaks : dict[int, np.ndarray]
            Peak indices (e.g. from :meth:`detect_spikes`).
        half_width : int
            Waveform half-width in samples for extraction.
        exclude_channels : list of int, optional
            Channels to exclude (e.g., noisy channels).
        t_search_start_ms : float, optional
            Start searching for T-end this many ms after Q. Default 50 ms.
        t_search_end_ms : float, optional
            Stop searching for T-end this many ms after Q. Default 500 ms.
        derivative_threshold : float, optional
            Fraction of max derivative for T-end detection. Default 0.1.
        baseline_threshold : float, optional
            Fraction of T-wave amplitude for baseline crossing. Default 0.1.
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
            - **qt_avg_min/max/mean/median/std_ms** : average of all methods
        """
        waves, _ = self.extract_waveforms(
            peaks, half_width=half_width, exclude_channels=exclude_channels
        )
        return _compute_qt_intervals(
            waves,
            self.sample_rate,
            exclude_channels=exclude_channels,
            t_search_start_ms=t_search_start_ms,
            t_search_end_ms=t_search_end_ms,
            derivative_threshold=derivative_threshold,
            baseline_threshold=baseline_threshold,
            smoothing_ms=smoothing_ms,
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_channels(self, **kwargs) -> Any:
        """Quick overview plot of all channels.

        Keyword arguments are forwarded to
        :func:`~intan_reader.visualization.plot_channels`. Common options:

        - ``start_seconds`` (float) — display window start, default 2
        - ``duration_seconds`` (float) — display window length, default 20
        - ``y_min`` / ``y_max`` — y-axis limits in µV
        - ``show`` (bool) — call ``plt.show()``, default True

        Returns
        -------
        (fig, axes)
        """
        from intan_reader.visualization import plot_channels

        kwargs.setdefault("sample_rate", self.sample_rate)
        return plot_channels(self.amplifier_data, **kwargs)

    def plot_waveforms(
        self,
        peaks: Dict[int, np.ndarray],
        *,
        half_width: int = 5000,
        channels: Optional[List[int]] = None,
        exclude_channels: Optional[List[int]] = None,
        **kwargs,
    ) -> Any:
        """Extract and plot spike waveforms.

        Parameters
        ----------
        peaks : dict[int, np.ndarray]
            Peak indices (e.g. from :meth:`detect_spikes`).
        half_width : int
            Waveform half-width in samples.
        channels : list of int, optional
            Channels to plot. ``None`` → all channels with spikes.
        exclude_channels : list of int, optional
            Channels to exclude from plotting (e.g., noisy channels from
            :meth:`detect_noisy_channels`).

        Returns
        -------
        (fig, axes)
        """
        from intan_reader.visualization import plot_waveforms

        waves, avg = self.extract_waveforms(peaks, half_width=half_width)
        kwargs.setdefault("sample_rate", self.sample_rate)
        return plot_waveforms(
            waves, avg, channels=channels, exclude_channels=exclude_channels, **kwargs
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Recording(channels={self.num_channels}, "
            f"samples={self.num_samples}, "
            f"duration={self.duration_seconds:.2f}s, "
            f"sample_rate={self.sample_rate:.0f} Hz)"
        )
