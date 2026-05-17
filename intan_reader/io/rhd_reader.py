"""
High-level reader for Intan Technologies RHD2000 data files.

Wraps the low-level header and data-block parsers into a single
:func:`read_rhd_file` entry point that returns a fully-parsed result
dictionary with scaled voltage data and time vectors.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from intan_reader.io.rhd_header import read_header
from intan_reader.io.rhd_data_block import get_bytes_per_data_block, read_one_data_block

logger = logging.getLogger(__name__)


def read_rhd_file(
    filepath: Union[str, Path],
    *,
    apply_notch: bool = True,
    notch_freq: Optional[float] = None,
    highpass_cutoff: Optional[float] = None,
    lowpass_cutoff: Optional[float] = None,
    sample_rate_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Read an Intan RHD2000 data file and return scaled data.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.rhd`` file.
    apply_notch : bool, optional
        If ``True`` (default) and a notch filter was enabled during
        acquisition, apply the same notch filter to the amplifier data.
        Ignored when *notch_freq* is explicitly set.
    notch_freq : float or None, optional
        Notch filter frequency in Hz (e.g. 50 or 60).  When set, this
        **overrides** both the file header setting and *apply_notch*.
        Pass ``0`` to explicitly disable the notch filter even if one was
        active during acquisition.
    highpass_cutoff : float or None, optional
        If given, apply a 4th-order Butterworth high-pass filter at this
        frequency (Hz) after loading.
    lowpass_cutoff : float or None, optional
        If given, apply a 4th-order Butterworth low-pass filter at this
        frequency (Hz) after loading.
    sample_rate_override : float or None, optional
        Override the sample rate read from the file header.  Useful when
        the recording was made at a non-standard rate (e.g. 10 kS/s) and
        the header value needs correction.  This affects all derived time
        vectors and any subsequent filtering.

    Returns
    -------
    dict
        A dictionary with the following possible keys (presence depends on
        what was recorded):

        - **amplifier_data** – ``(n_channels, n_samples)`` array in µV
        - **t_amplifier** – time vector in seconds
        - **aux_input_data** – auxiliary input voltages in V
        - **supply_voltage_data** – supply voltages in V
        - **board_adc_data** – board ADC voltages in V
        - **board_dig_in_data** – digital input states (bool)
        - **board_dig_out_data** – digital output states (bool)
        - **temp_sensor_data** – temperature in °C
        - **amplifier_channels** – list of channel metadata dicts
        - **frequency_parameters** – sampling / filter parameters
        - **notes** – user notes from acquisition
        - **header** – the raw parsed header dict
        - **sample_rate** – amplifier sample rate in Hz

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file is not a valid RHD2000 file or has unexpected size.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"RHD file not found: {filepath}")

    tic = time.time()
    filesize = filepath.stat().st_size

    with open(filepath, "rb") as fid:
        header = read_header(fid)

        logger.info(
            "Channels — amplifier: %d, aux: %d, ADC: %d, dig-in: %d, dig-out: %d",
            header["num_amplifier_channels"],
            header["num_aux_input_channels"],
            header["num_board_adc_channels"],
            header["num_board_dig_in_channels"],
            header["num_board_dig_out_channels"],
        )

        # --- Determine number of data blocks --------------------------------
        bytes_per_block = get_bytes_per_data_block(header)
        bytes_remaining = filesize - fid.tell()

        if bytes_remaining == 0:
            logger.warning("Header-only file — no data present.")
            return _build_result(header, data=None)

        if bytes_remaining % bytes_per_block != 0:
            raise ValueError(
                f"File size mismatch: {bytes_remaining} leftover bytes are not "
                f"a whole number of {bytes_per_block}-byte data blocks."
            )

        num_data_blocks = int(bytes_remaining / bytes_per_block)
        n = header["num_samples_per_data_block"]

        num_amp_samples = n * num_data_blocks
        num_aux_samples = (n // 4) * num_data_blocks
        num_sv_samples = num_data_blocks
        num_adc_samples = n * num_data_blocks
        num_dig_samples = n * num_data_blocks

        record_time = num_amp_samples / header["sample_rate"]
        logger.info(
            "%.3f s of data at %.2f kS/s (%d blocks)",
            record_time,
            header["sample_rate"] / 1000,
            num_data_blocks,
        )

        # --- Pre-allocate arrays --------------------------------------------
        t_dtype = (
            np.int32
            if (header["version"]["major"] == 1 and header["version"]["minor"] >= 2)
            or header["version"]["major"] > 1
            else np.uint32
        )
        data: Dict[str, np.ndarray] = {
            "t_amplifier": np.zeros(num_amp_samples, dtype=t_dtype),
            "amplifier_data": np.zeros(
                (header["num_amplifier_channels"], num_amp_samples), dtype=np.uint16
            ),
            "aux_input_data": np.zeros(
                (header["num_aux_input_channels"], num_aux_samples), dtype=np.uint16
            ),
            "supply_voltage_data": np.zeros(
                (header["num_supply_voltage_channels"], num_sv_samples), dtype=np.uint16
            ),
            "temp_sensor_data": np.zeros(
                (header["num_temp_sensor_channels"], num_sv_samples), dtype=np.uint16
            ),
            "board_adc_data": np.zeros(
                (header["num_board_adc_channels"], num_adc_samples), dtype=np.uint16
            ),
            "board_dig_in_data": np.zeros(
                (header["num_board_dig_in_channels"], num_dig_samples), dtype=bool
            ),
            "board_dig_in_raw": np.zeros(num_dig_samples, dtype=np.uint16),
            "board_dig_out_data": np.zeros(
                (header["num_board_dig_out_channels"], num_dig_samples), dtype=bool
            ),
            "board_dig_out_raw": np.zeros(num_dig_samples, dtype=np.uint16),
        }

        # --- Read all blocks ------------------------------------------------
        indices = {
            "amplifier": 0,
            "aux_input": 0,
            "supply_voltage": 0,
            "board_adc": 0,
            "board_dig_in": 0,
            "board_dig_out": 0,
        }

        for i in range(num_data_blocks):
            read_one_data_block(data, header, indices, fid)

            indices["amplifier"] += n
            indices["aux_input"] += n // 4
            indices["supply_voltage"] += 1
            indices["board_adc"] += n
            indices["board_dig_in"] += n
            indices["board_dig_out"] += n

            if (i + 1) % max(1, num_data_blocks // 10) == 0:
                logger.debug(
                    "Reading: %d%%", int(100 * (i + 1) / num_data_blocks)
                )

        # Verify we consumed exactly the right number of bytes.
        if filesize - fid.tell() != 0:
            raise ValueError("End of file not reached — data may be corrupt.")

    # --- Sample-rate override ------------------------------------------------
    if sample_rate_override is not None:
        logger.info(
            "Overriding sample rate: %.0f -> %.0f Hz",
            header["sample_rate"],
            sample_rate_override,
        )
        header["sample_rate"] = sample_rate_override
        header["frequency_parameters"]["amplifier_sample_rate"] = sample_rate_override
        header["frequency_parameters"]["aux_input_sample_rate"] = sample_rate_override / 4
        header["frequency_parameters"]["supply_voltage_sample_rate"] = (
            sample_rate_override / header["num_samples_per_data_block"]
        )
        header["frequency_parameters"]["board_adc_sample_rate"] = sample_rate_override
        header["frequency_parameters"]["board_dig_in_sample_rate"] = sample_rate_override

    # --- Post-processing: scale & extract ----------------------------------
    _scale_data(data, header)
    _extract_digital_channels(data, header)
    _compute_time_vectors(data, header)

    # --- Filtering ---------------------------------------------------------
    # Determine effective notch frequency
    if notch_freq is not None:
        effective_notch = notch_freq
    elif apply_notch:
        effective_notch = header["notch_filter_frequency"]
    else:
        effective_notch = 0

    if effective_notch > 0:
        _apply_notch_filter(data, header, notch_freq=effective_notch)

    if highpass_cutoff is not None:
        _apply_highpass_filter(data, header, cutoff=highpass_cutoff)

    if lowpass_cutoff is not None:
        _apply_lowpass_filter(data, header, cutoff=lowpass_cutoff)

    elapsed = time.time() - tic
    logger.info("File read complete in %.1f s", elapsed)

    return _build_result(header, data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scale_data(data: Dict[str, np.ndarray], header: Dict[str, Any]) -> None:
    """Convert raw uint16 values to physical units (in-place)."""
    # Amplifier: µV
    data["amplifier_data"] = 0.195 * (data["amplifier_data"].astype(np.int32) - 32768)
    # Auxiliary inputs: V
    data["aux_input_data"] = 37.4e-6 * data["aux_input_data"].astype(np.float64)
    # Supply voltage: V
    data["supply_voltage_data"] = 74.8e-6 * data["supply_voltage_data"].astype(np.float64)
    # Board ADC: V (scaling depends on eval board mode)
    mode = header["eval_board_mode"]
    if mode == 1:
        data["board_adc_data"] = 152.59e-6 * (
            data["board_adc_data"].astype(np.int32) - 32768
        )
    elif mode == 13:
        data["board_adc_data"] = 312.5e-6 * (
            data["board_adc_data"].astype(np.int32) - 32768
        )
    else:
        data["board_adc_data"] = 50.354e-6 * data["board_adc_data"].astype(np.float64)
    # Temperature: °C
    data["temp_sensor_data"] = 0.01 * data["temp_sensor_data"].astype(np.float64)


def _extract_digital_channels(
    data: Dict[str, np.ndarray], header: Dict[str, Any]
) -> None:
    """Unpack per-channel booleans from raw digital words."""
    for i in range(header["num_board_dig_in_channels"]):
        mask = 1 << header["board_dig_in_channels"][i]["native_order"]
        data["board_dig_in_data"][i, :] = (data["board_dig_in_raw"] & mask) != 0

    for i in range(header["num_board_dig_out_channels"]):
        mask = 1 << header["board_dig_out_channels"][i]["native_order"]
        data["board_dig_out_data"][i, :] = (data["board_dig_out_raw"] & mask) != 0


def _compute_time_vectors(
    data: Dict[str, np.ndarray], header: Dict[str, Any]
) -> None:
    """Derive time vectors in seconds from the timestamp array."""
    num_gaps = np.sum(np.diff(data["t_amplifier"]) != 1)
    if num_gaps > 0:
        logger.warning("%d gaps detected in timestamp data.", num_gaps)

    sr = header["sample_rate"]
    n = header["num_samples_per_data_block"]

    data["t_amplifier"] = data["t_amplifier"].astype(np.float64) / sr
    data["t_aux_input"] = data["t_amplifier"][::4]
    data["t_supply_voltage"] = data["t_amplifier"][::n]
    data["t_board_adc"] = data["t_amplifier"]
    data["t_dig"] = data["t_amplifier"]
    data["t_temp_sensor"] = data["t_supply_voltage"]


def _apply_notch_filter(
    data: Dict[str, np.ndarray],
    header: Dict[str, Any],
    *,
    notch_freq: Optional[float] = None,
) -> None:
    """Apply a notch filter to amplifier data."""
    from intan_reader.filters import notch

    freq = notch_freq if notch_freq is not None else header["notch_filter_frequency"]
    sr = header["sample_rate"]
    logger.info("Applying notch filter at %.1f Hz (sr=%.0f Hz)", freq, sr)
    data["amplifier_data"] = notch(data["amplifier_data"], sr, freq)


def _apply_highpass_filter(
    data: Dict[str, np.ndarray],
    header: Dict[str, Any],
    *,
    cutoff: float,
    order: int = 4,
) -> None:
    """Apply a high-pass Butterworth filter to amplifier data."""
    from intan_reader.filters import highpass

    sr = header["sample_rate"]
    logger.info("Applying high-pass filter at %.1f Hz (sr=%.0f Hz)", cutoff, sr)
    data["amplifier_data"] = highpass(data["amplifier_data"], sr, cutoff, order)


def _apply_lowpass_filter(
    data: Dict[str, np.ndarray],
    header: Dict[str, Any],
    *,
    cutoff: float,
    order: int = 4,
) -> None:
    """Apply a low-pass Butterworth filter to amplifier data."""
    from intan_reader.filters import lowpass

    sr = header["sample_rate"]
    logger.info("Applying low-pass filter at %.1f Hz (sr=%.0f Hz)", cutoff, sr)
    data["amplifier_data"] = lowpass(data["amplifier_data"], sr, cutoff, order)


def _build_result(
    header: Dict[str, Any],
    data: Optional[Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Assemble the public result dictionary."""
    result: Dict[str, Any] = {
        "header": header,
        "sample_rate": header["sample_rate"],
        "frequency_parameters": header["frequency_parameters"],
        "notes": header["notes"],
    }

    if header["version"]["major"] > 1:
        result["reference_channel"] = header.get("reference_channel")

    if data is None:
        return result

    # Copy data arrays into result
    _CHANNEL_KEYS = [
        ("amplifier_channels", "amplifier_data", "t_amplifier", "spike_triggers"),
        ("aux_input_channels", "aux_input_data", "t_aux_input", None),
        ("supply_voltage_channels", "supply_voltage_data", "t_supply_voltage", None),
        ("board_adc_channels", "board_adc_data", "t_board_adc", None),
        ("board_dig_in_channels", "board_dig_in_data", "t_dig", None),
        ("board_dig_out_channels", "board_dig_out_data", "t_dig", None),
    ]

    for ch_key, data_key, time_key, extra_key in _CHANNEL_KEYS:
        if header[f"num_{ch_key}"] > 0:
            result[ch_key] = header[ch_key]
            if data_key in data:
                result[data_key] = data[data_key]
            if time_key in data:
                result[time_key] = data[time_key]
            if extra_key is not None and extra_key in header:
                result[extra_key] = header[extra_key]

    if header["num_temp_sensor_channels"] > 0:
        result["temp_sensor_data"] = data.get("temp_sensor_data")
        result["t_temp_sensor"] = data.get("t_temp_sensor")

    return result
