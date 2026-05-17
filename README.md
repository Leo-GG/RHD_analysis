# intan-reader

A Python library for reading and processing Intan Technologies RHD2000 electrophysiology data, with specialized support for cardiac MEA (multi-electrode array) recordings.

## Features

- **Load RHD files** — parse binary headers, read amplifier / auxiliary / digital data, scale to physical units (µV, V, °C)
- **Merge recordings** — concatenate multi-file sessions by base name
- **Signal filtering** — zero-phase notch, high-pass, low-pass, and band-pass (Butterworth via SciPy)
- **Artifact detection** — envelope-based or threshold-based, per channel
- **Noise detection** — identify noisy channels using Hartigan's dip test
- **Spike detection** — negative-going peak detection with amplitude and waveform-shape filtering
- **Waveform extraction** — cut and average spike snippets
- **Spike statistics** — per-channel amplitude, frequency, ISI, and waveform stability metrics
- **Cardiac QT analysis** — four methods for T-wave end detection (derivative, baseline, tangent, Gaussian fit)
- **Visualization** — quick 8×8 channel grid, spike waveform overlay plots, QT detection plots

## Installation

```bash
pip install -e .              # editable install
# or, with all optional dependencies:
pip install -e ".[all]"
```

### Dependencies

| Package    | Minimum | Purpose                    |
|------------|---------|----------------------------|
| numpy      | 1.22    | Array operations           |
| scipy      | 1.9     | Filtering, peaks, fitting  |
| diptest    | 0.7     | Noise detection            |
| pandas     | 2.0     | Statistics DataFrames      |
| matplotlib | 3.5     | Visualization (optional)   |

## Quick Start

### Load a single file

```python
from intan_reader import Recording

rec = Recording.from_file("data/experiment_001.rhd")
print(rec)
# Recording(channels=64, samples=1200000, duration=60.00s, sample_rate=20000 Hz)
```

### Load and merge multiple files

```python
rec = Recording.from_folder("recordings/", "Exp1_condition_3")
# Merges all files in recordings/ whose name contains "Exp1_condition_3"
```

### Load with custom filtering

```python
rec = Recording.from_file(
    "data/experiment.rhd",
    notch_freq=50,           # 50 Hz mains removal
    highpass_cutoff=300,     # high-pass at 300 Hz
    lowpass_cutoff=3000,     # low-pass at 3000 Hz
)
```

### Detect artifacts and spikes

```python
# Detect artifacts
artifacts = rec.detect_artifacts(method="envelope", n_std=3.0)

# Detect spikes
raw_peaks, filtered_peaks = rec.detect_spikes(threshold=3.5)

# Extract waveform snippets
waveforms, averages = rec.extract_waveforms(filtered_peaks)
```

### Detect noisy channels

```python
noise_result = rec.detect_noisy_channels(dip_threshold=0.05)
noisy_channels = [i for i, is_noisy in enumerate(noise_result["is_noisy"]) if is_noisy]
```

### Compute spike statistics

```python
stats_df = rec.compute_spike_statistics(
    filtered_peaks,
    exclude_channels=noisy_channels
)
print(stats_df)
# Returns DataFrame with: n_spikes, freq_hz, amp_min/max/mean/median/std,
# isi_min/max/mean/median/std_ms, wf_dev_min/max/mean/median
```

### Cardiac QT interval analysis

```python
qt_df = rec.compute_qt_intervals(
    filtered_peaks,
    exclude_channels=noisy_channels,
    t_search_start_ms=50,
    t_search_end_ms=500,
)
# Returns DataFrame with QT stats from 4 methods:
# - Derivative threshold
# - Baseline crossing
# - Tangent (steepest descent)
# - Gaussian fit (mu + 2*sigma)
```

### Visualize QT detection

```python
from intan_reader.processing import plot_qt_detection

# Get a single waveform
waves, _ = rec.extract_waveforms(filtered_peaks, half_width=5000)
waveform = waves[channel_id][beat_index]

# Plot with all 4 detection methods
plot_qt_detection(waveform, rec.sample_rate)
```

### Visualize channels and waveforms

```python
rec.plot_channels(start_seconds=2, duration_seconds=20)
rec.plot_waveforms(filtered_peaks, exclude_channels=noisy_channels)
```

## Project Structure

```
intan_reader/
├── __init__.py           # Public API
├── recording.py          # Recording class — main entry point
├── filters.py            # notch, highpass, lowpass, bandpass
├── visualization.py      # plot_channels, plot_waveforms
├── io/
│   ├── rhd_reader.py     # High-level file reader
│   ├── rhd_header.py     # Binary header parser
│   ├── rhd_data_block.py # Data block reader
│   └── qstring.py        # Qt QString binary reader
└── processing/
    ├── artifacts.py      # Artifact detection
    ├── spikes.py         # Peak detection, filtering, waveform extraction, statistics
    ├── noise_detection.py # Noisy channel detection (dip test)
    └── cardiac.py        # QT interval estimation (4 methods)
```

## API Reference

### Recording class

| Method | Description |
|--------|-------------|
| `from_file(path)` | Load a single RHD file |
| `from_folder(folder, pattern)` | Load and merge files matching pattern |
| `apply_notch(freq)` | Apply notch filter |
| `apply_highpass(cutoff)` | Apply high-pass filter |
| `apply_bandpass(low, high)` | Apply band-pass filter |
| `detect_artifacts(method, ...)` | Detect artifact regions |
| `detect_noisy_channels(...)` | Identify noisy channels via dip test |
| `detect_spikes(threshold, ...)` | Detect spike peaks |
| `extract_waveforms(peaks, ...)` | Extract waveform snippets |
| `compute_spike_statistics(...)` | Compute per-channel spike metrics |
| `compute_qt_intervals(...)` | Compute QT intervals (cardiac) |
| `plot_channels(...)` | Plot channel overview |
| `plot_waveforms(...)` | Plot spike waveforms |

### Processing functions

```python
from intan_reader.processing import (
    detect_artifacts,
    detect_peaks,
    filter_peaks,
    extract_waveforms,
    compute_spike_statistics,
    detect_noisy_channels,
    estimate_qt_interval,
    compute_qt_intervals,
    plot_qt_detection,
)
```

## License

MIT
