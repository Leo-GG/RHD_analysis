"""
Low-level routines for reading individual data blocks from RHD2000 files.

Each data block contains either 60 (v1.x) or 128 (v2.x) amplifier samples
plus associated auxiliary, ADC, and digital I/O samples.
"""

from __future__ import annotations

import struct
from typing import Any, Dict

import numpy as np


def get_bytes_per_data_block(header: Dict[str, Any]) -> int:
    """Calculate the byte size of one data block given the file header.

    Parameters
    ----------
    header : dict
        Parsed RHD header (from :func:`intan_reader.io.rhd_header.read_header`).

    Returns
    -------
    int
        Number of bytes per data block.
    """
    n = header["num_samples_per_data_block"]

    # Timestamps (4 bytes each)
    total = n * 4
    # Amplifier channels (2 bytes per sample per channel)
    total += n * 2 * header["num_amplifier_channels"]
    # Auxiliary inputs sampled 4× slower
    total += (n // 4) * 2 * header["num_aux_input_channels"]
    # Supply voltage sampled once per block
    total += 1 * 2 * header["num_supply_voltage_channels"]
    # Board ADC channels at full rate
    total += n * 2 * header["num_board_adc_channels"]
    # Digital inputs (one 16-bit word per sample if any channels exist)
    if header["num_board_dig_in_channels"] > 0:
        total += n * 2
    # Digital outputs
    if header["num_board_dig_out_channels"] > 0:
        total += n * 2
    # Temperature sensors sampled once per block
    if header["num_temp_sensor_channels"] > 0:
        total += 1 * 2 * header["num_temp_sensor_channels"]

    return total


def read_one_data_block(
    data: Dict[str, np.ndarray],
    header: Dict[str, Any],
    indices: Dict[str, int],
    fid,
) -> None:
    """Read one data block from *fid* into pre-allocated *data* arrays.

    The caller must advance the values in *indices* after each call.

    Parameters
    ----------
    data : dict of np.ndarray
        Pre-allocated arrays keyed by data type (``'amplifier_data'``, etc.).
    header : dict
        Parsed RHD header.
    indices : dict of int
        Current write positions for each data type.
    fid : BinaryIO
        Open file handle positioned at the start of a data block.
    """
    n = header["num_samples_per_data_block"]

    # --- Timestamps ------------------------------------------------------
    if (header["version"]["major"] == 1 and header["version"]["minor"] >= 2) or (
        header["version"]["major"] > 1
    ):
        fmt = "<" + "i" * n
    else:
        fmt = "<" + "I" * n
    data["t_amplifier"][indices["amplifier"] : indices["amplifier"] + n] = (
        np.array(struct.unpack(fmt, fid.read(4 * n)))
    )

    # --- Amplifier data --------------------------------------------------
    if header["num_amplifier_channels"] > 0:
        count = n * header["num_amplifier_channels"]
        tmp = np.fromfile(fid, dtype="uint16", count=count)
        data["amplifier_data"][
            :, indices["amplifier"] : indices["amplifier"] + n
        ] = tmp.reshape(header["num_amplifier_channels"], n)

    # --- Auxiliary input data --------------------------------------------
    if header["num_aux_input_channels"] > 0:
        aux_n = n // 4
        count = aux_n * header["num_aux_input_channels"]
        tmp = np.fromfile(fid, dtype="uint16", count=count)
        data["aux_input_data"][
            :, indices["aux_input"] : indices["aux_input"] + aux_n
        ] = tmp.reshape(header["num_aux_input_channels"], aux_n)

    # --- Supply voltage data ---------------------------------------------
    if header["num_supply_voltage_channels"] > 0:
        count = header["num_supply_voltage_channels"]
        tmp = np.fromfile(fid, dtype="uint16", count=count)
        data["supply_voltage_data"][
            :, indices["supply_voltage"] : indices["supply_voltage"] + 1
        ] = tmp.reshape(count, 1)

    # --- Temperature sensor data -----------------------------------------
    if header["num_temp_sensor_channels"] > 0:
        count = header["num_temp_sensor_channels"]
        tmp = np.fromfile(fid, dtype="uint16", count=count)
        data["temp_sensor_data"][
            :, indices["supply_voltage"] : indices["supply_voltage"] + 1
        ] = tmp.reshape(count, 1)

    # --- Board ADC data --------------------------------------------------
    if header["num_board_adc_channels"] > 0:
        count = n * header["num_board_adc_channels"]
        tmp = np.fromfile(fid, dtype="uint16", count=count)
        data["board_adc_data"][
            :, indices["board_adc"] : indices["board_adc"] + n
        ] = tmp.reshape(header["num_board_adc_channels"], n)

    # --- Board digital input data ----------------------------------------
    if header["num_board_dig_in_channels"] > 0:
        data["board_dig_in_raw"][
            indices["board_dig_in"] : indices["board_dig_in"] + n
        ] = np.array(struct.unpack("<" + "H" * n, fid.read(2 * n)))

    # --- Board digital output data ---------------------------------------
    if header["num_board_dig_out_channels"] > 0:
        data["board_dig_out_raw"][
            indices["board_dig_out"] : indices["board_dig_out"] + n
        ] = np.array(struct.unpack("<" + "H" * n, fid.read(2 * n)))
