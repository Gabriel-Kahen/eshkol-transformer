"""Byte-level V1 oracle; provenance and live-owner behavior require the compiled gate."""
import argparse
import hashlib
from pathlib import Path
import struct

from tests.g3r.validate_trained_pair_artifacts import verify as verify_pair


SIZE = 6292
C2_SHA = "9ec2a1d40ebd43e57d1e8b75fa8f6d96cd4d17abc1002f70da04d20144ee78df"
COMPANION_SHA = "75fec71697cf50556228481b0a2a30f634ce8991a7f6db4ad7fbdc700f1cfbf2"
SHAPES = ((4, 4),) * 4 + ((4, 8), (8, 4)) + ((4, 0),) * 4 + (
    (256, 4),) + ((4, 0),) * 2 + ((2, 4),)


def decode(data: bytes, c2: bytes, companion: bytes, destination: list) -> None:
    """Publish one immutable copy only after every structural and pair check."""
    if len(destination) != 1 or len(data) != SIZE or len(companion) != 288:
        raise ValueError("transcript extents")
    verify_pair(c2, c2, companion, 65, budget=1)
    if (data[:8] != b"G3RTV1\0\0" or
            struct.unpack_from("<II", data, 8) != (1, SIZE) or
            data[16:48] != hashlib.sha256(c2).digest() or
            data[48:80] != hashlib.sha256(companion).digest() or
            struct.unpack_from("<4I", data, 80) != (65, 2, 14, 0) or
            data[6260:] != hashlib.sha256(data[:6260]).digest()):
        raise ValueError("transcript header, pair or checksum")
    offset = 96
    for index, (dim0, dim1) in enumerate(SHAPES):
        rank = 1 if dim1 == 0 else 2
        if struct.unpack_from("<4I", data, offset) != (index, rank, dim0, dim1):
            raise ValueError("parameter roster")
        offset += 16 + 4 * dim0 * (dim1 or 1)
    if offset != 5056:
        raise ValueError("parameter extent")
    if (struct.unpack_from("<2I", data, 6144) != (1, 1) or
            struct.unpack_from("<I", data, 6152)[0] > 255 or
            struct.unpack_from("<2I", data, 6252) != (1, 2) or
            struct.unpack_from("<4q", data, 6112) !=
            struct.unpack_from("<4q", companion, 88)):
        raise ValueError("P1/G1 trace")
    destination[0] = bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    args = parser.parse_args()
    trained, saved = args.trained.read_bytes(), args.saved.read_bytes()
    companion, transcript = args.companion.read_bytes(), args.transcript.read_bytes()
    if trained != saved:
        raise ValueError("saved C2 bytes changed")
    if (hashlib.sha256(saved).hexdigest() != C2_SHA or
            hashlib.sha256(companion).hexdigest() != COMPANION_SHA):
        raise ValueError("not the authenticated budget-one C2 pair")
    destination = [None]
    decode(transcript, saved, companion, destination)
    if destination[0] != transcript:
        raise ValueError("decoded transcript changed")
    print("G3-R TRAINED TRANSCRIPT BYTES PASS")


if __name__ == "__main__":
    main()
