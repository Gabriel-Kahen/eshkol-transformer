#!/usr/bin/env python3
"""Independent exact-wire fixture for the private O2 staging adapter."""

from pathlib import Path
import sys

from test_checkpoint_format import make_fixture, make_optimizer


def main() -> int:
    metadata_path, payload_path, fixture_path = map(Path, sys.argv[1:4])
    metadata_raw, payload, offsets = make_optimizer()
    metadata = bytearray(metadata_raw)
    record = offsets["record"]
    metadata[record + 56:record + 120] = bytes(64)
    metadata_path.write_bytes(metadata)
    payload_path.write_bytes(payload)
    fixture, fixture_offsets = make_fixture()
    optimizer = fixture_offsets["optimizer"]
    fixture[optimizer:optimizer + len(metadata)] = metadata
    fixture_path.write_bytes(fixture)
    assert metadata[record + 56:record + 120] == bytes(64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
