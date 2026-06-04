"""
intan_reader.io - Low-level I/O for Intan Technologies RHD2000 data files.

This subpackage handles binary file parsing, header reading, and raw data
extraction from RHD files produced by the Intan Recording Controller or
Evaluation Board GUI.
"""

from intan_reader.io.rhd_reader import read_rhd_file

__all__ = ["read_rhd_file"]
