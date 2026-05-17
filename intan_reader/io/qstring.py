"""
Qt-style QString binary reader for Intan RHD file headers.

The RHD file format stores strings in Qt's QString serialisation format:
a 32-bit unsigned length (in bytes) followed by UTF-16LE code units.
A length of 0xFFFFFFFF represents a null (empty) string.
"""

from __future__ import annotations

import os
import struct
from typing import BinaryIO


def read_qstring(fid: BinaryIO) -> str:
    """Read a single Qt-style QString from an open binary file.

    Parameters
    ----------
    fid : BinaryIO
        An open file handle positioned at the start of a QString.

    Returns
    -------
    str
        The decoded Python string. Returns ``""`` for null QStrings.

    Raises
    ------
    ValueError
        If the encoded length exceeds the remaining file size.
    """
    (length,) = struct.unpack("<I", fid.read(4))

    # 0xFFFFFFFF signals a null QString
    if length == 0xFFFFFFFF:
        return ""

    remaining = os.fstat(fid.fileno()).st_size - fid.tell() + 1
    if length > remaining:
        raise ValueError(
            f"QString length ({length} bytes) exceeds remaining file size "
            f"({remaining} bytes)."
        )

    # Length is in bytes; each UTF-16 code unit is 2 bytes.
    n_chars = length // 2
    data = struct.unpack(f"<{n_chars}H", fid.read(2 * n_chars))
    return "".join(chr(c) for c in data)
