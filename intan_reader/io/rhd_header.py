"""
RHD2000 file header parser.

Reads the binary header from an Intan Technologies RHD2000 data file,
extracting version info, sampling parameters, frequency settings,
channel metadata, and spike trigger configuration.

Based on the Intan file-format specification (versions 1.x and 2.x).
"""

from __future__ import annotations

import logging
import struct
from typing import Any, BinaryIO, Dict, List

from intan_reader.io.qstring import read_qstring

logger = logging.getLogger(__name__)

# Magic number that identifies a valid RHD2000 file.
RHD_MAGIC_NUMBER = 0xC6912702


def read_header(fid: BinaryIO) -> Dict[str, Any]:
    """Parse the RHD2000 binary header from an open file.

    Parameters
    ----------
    fid : BinaryIO
        A file handle opened in binary-read mode (``'rb'``), positioned at
        byte 0 of an RHD2000 file.

    Returns
    -------
    dict
        A dictionary containing all header fields. Key sections include:

        - ``version`` – ``{"major": int, "minor": int}``
        - ``sample_rate`` – amplifier sampling rate in Hz
        - ``frequency_parameters`` – bandwidth / DSP / notch settings
        - ``amplifier_channels``, ``aux_input_channels``, etc. – channel lists
        - ``num_amplifier_channels``, ``num_aux_input_channels``, etc. – counts
        - ``num_samples_per_data_block`` – 60 (v1.x) or 128 (v2.x)
        - ``notes`` – user notes stored during acquisition
        - ``eval_board_mode`` – evaluation board operating mode

    Raises
    ------
    ValueError
        If the magic number does not match the RHD2000 format.
    """
    (magic_number,) = struct.unpack("<I", fid.read(4))
    if magic_number != RHD_MAGIC_NUMBER:
        raise ValueError(
            f"Unrecognised file type (magic number 0x{magic_number:08X}). "
            "Expected an Intan Technologies RHD2000 data file."
        )

    header: Dict[str, Any] = {}

    # -- Version ----------------------------------------------------------
    version: Dict[str, int] = {}
    version["major"], version["minor"] = struct.unpack("<hh", fid.read(4))
    header["version"] = version
    logger.info(
        "RHD2000 file version %d.%d", version["major"], version["minor"]
    )

    # -- Sampling / frequency parameters ----------------------------------
    freq: Dict[str, Any] = {}
    (header["sample_rate"],) = struct.unpack("<f", fid.read(4))

    (
        freq["dsp_enabled"],
        freq["actual_dsp_cutoff_frequency"],
        freq["actual_lower_bandwidth"],
        freq["actual_upper_bandwidth"],
        freq["desired_dsp_cutoff_frequency"],
        freq["desired_lower_bandwidth"],
        freq["desired_upper_bandwidth"],
    ) = struct.unpack("<hffffff", fid.read(26))

    # Software notch filter (0 = none, 1 = 50 Hz, 2 = 60 Hz)
    (notch_filter_mode,) = struct.unpack("<h", fid.read(2))
    header["notch_filter_frequency"] = {0: 0, 1: 50, 2: 60}.get(
        notch_filter_mode, 0
    )
    freq["notch_filter_frequency"] = header["notch_filter_frequency"]

    (
        freq["desired_impedance_test_frequency"],
        freq["actual_impedance_test_frequency"],
    ) = struct.unpack("<ff", fid.read(8))

    # -- Notes ------------------------------------------------------------
    header["notes"] = {
        "note1": read_qstring(fid),
        "note2": read_qstring(fid),
        "note3": read_qstring(fid),
    }

    # -- Temperature sensors (v1.1+) -------------------------------------
    header["num_temp_sensor_channels"] = 0
    if (version["major"] == 1 and version["minor"] >= 1) or version["major"] > 1:
        (header["num_temp_sensor_channels"],) = struct.unpack("<h", fid.read(2))

    # -- Eval board mode (v1.3+) ------------------------------------------
    header["eval_board_mode"] = 0
    if (version["major"] == 1 and version["minor"] >= 3) or version["major"] > 1:
        (header["eval_board_mode"],) = struct.unpack("<h", fid.read(2))

    # -- Samples per data block -------------------------------------------
    header["num_samples_per_data_block"] = 60
    if version["major"] > 1:
        header["reference_channel"] = read_qstring(fid)
        header["num_samples_per_data_block"] = 128

    # -- Derived sample rates ---------------------------------------------
    freq["amplifier_sample_rate"] = header["sample_rate"]
    freq["aux_input_sample_rate"] = header["sample_rate"] / 4
    freq["supply_voltage_sample_rate"] = (
        header["sample_rate"] / header["num_samples_per_data_block"]
    )
    freq["board_adc_sample_rate"] = header["sample_rate"]
    freq["board_dig_in_sample_rate"] = header["sample_rate"]
    header["frequency_parameters"] = freq

    # -- Channel lists ----------------------------------------------------
    header["spike_triggers"]: List[Dict] = []
    header["amplifier_channels"]: List[Dict] = []
    header["aux_input_channels"]: List[Dict] = []
    header["supply_voltage_channels"]: List[Dict] = []
    header["board_adc_channels"]: List[Dict] = []
    header["board_dig_in_channels"]: List[Dict] = []
    header["board_dig_out_channels"]: List[Dict] = []

    _SIGNAL_TYPE_MAP = {
        0: ("amplifier_channels", "spike_triggers"),
        1: ("aux_input_channels", None),
        2: ("supply_voltage_channels", None),
        3: ("board_adc_channels", None),
        4: ("board_dig_in_channels", None),
        5: ("board_dig_out_channels", None),
    }

    (number_of_signal_groups,) = struct.unpack("<h", fid.read(2))
    logger.debug("Signal groups in file: %d", number_of_signal_groups)

    for _ in range(number_of_signal_groups):
        signal_group_name = read_qstring(fid)
        signal_group_prefix = read_qstring(fid)
        (
            signal_group_enabled,
            signal_group_num_channels,
            _signal_group_num_amp_channels,
        ) = struct.unpack("<hhh", fid.read(6))

        if signal_group_num_channels <= 0 or signal_group_enabled <= 0:
            continue

        for _ in range(signal_group_num_channels):
            channel: Dict[str, Any] = {
                "port_name": signal_group_name,
                "port_prefix": signal_group_prefix,
            }
            channel["native_channel_name"] = read_qstring(fid)
            channel["custom_channel_name"] = read_qstring(fid)

            (
                channel["native_order"],
                channel["custom_order"],
                signal_type,
                channel_enabled,
                channel["chip_channel"],
                channel["board_stream"],
            ) = struct.unpack("<hhhhhh", fid.read(12))

            trigger: Dict[str, int] = {}
            (
                trigger["voltage_trigger_mode"],
                trigger["voltage_threshold"],
                trigger["digital_trigger_channel"],
                trigger["digital_edge_polarity"],
            ) = struct.unpack("<hhhh", fid.read(8))

            (
                channel["electrode_impedance_magnitude"],
                channel["electrode_impedance_phase"],
            ) = struct.unpack("<ff", fid.read(8))

            if not channel_enabled:
                continue

            if signal_type not in _SIGNAL_TYPE_MAP:
                raise ValueError(f"Unknown channel type: {signal_type}")

            ch_key, trig_key = _SIGNAL_TYPE_MAP[signal_type]
            header[ch_key].append(channel)
            if trig_key is not None:
                header[trig_key].append(trigger)

    # -- Channel counts ---------------------------------------------------
    header["num_amplifier_channels"] = len(header["amplifier_channels"])
    header["num_aux_input_channels"] = len(header["aux_input_channels"])
    header["num_supply_voltage_channels"] = len(header["supply_voltage_channels"])
    header["num_board_adc_channels"] = len(header["board_adc_channels"])
    header["num_board_dig_in_channels"] = len(header["board_dig_in_channels"])
    header["num_board_dig_out_channels"] = len(header["board_dig_out_channels"])

    return header
