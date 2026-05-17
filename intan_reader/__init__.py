"""
intan_reader - A Python library for reading and processing Intan Technologies RHD2000 data.

This package provides a clean API for:
- Loading RHD2000 electrophysiology recordings
- Merging multi-file recording sessions
- Signal filtering (notch, high-pass, low-pass, bandpass)
- Artifact detection and rejection
- Spike detection, filtering, and waveform extraction
- Quick visualization of multi-channel data

Quick Start
-----------
>>> from intan_reader import Recording
>>> rec = Recording.from_folder("recordings/", "Exp1_condition_3")
>>> rec.amplifier_data.shape
(64, 1200000)
>>> rec.detect_artifacts()
>>> peaks, filtered_peaks = rec.detect_spikes(threshold=3.5)
"""

from intan_reader.recording import Recording
from intan_reader.io import read_rhd_file
from intan_reader import filters
from intan_reader import processing

__version__ = "1.0.0"
__all__ = ["Recording", "read_rhd_file", "filters", "processing"]
