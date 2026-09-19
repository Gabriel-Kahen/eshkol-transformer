#!/usr/bin/env python3
"""Generate deterministic D1 resources for compiled D2 adversarial tests."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import struct

from tests.d2.test_resources import write_d1_resource


FINGERPRINT = "sha256:eshkol-bpe-tokenizer-v1:1866ebedd76bf7d0e8e111ab25603a99aee927ea341e1305e4bb2c145648eb72"
VOCAB = 263


def clone(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def rehash_file(path: Path) -> bytes:
    data = bytearray(path.read_bytes())
    data[-32:] = hashlib.sha256(data[:-32]).digest()
    path.write_bytes(data)
    return bytes(data[-32:])


def bind_shard(directory: Path, index: int) -> None:
    shard = directory / f"shard-{index:016d}.ets"
    shard_digest = rehash_file(shard)
    manifest = bytearray((directory / "manifest.etm").read_bytes())
    header_bytes = struct.unpack_from("<I", manifest, 12)[0]
    record = header_bytes + 56 * index
    manifest[record + 24 : record + 56] = shard_digest
    manifest[-32:] = hashlib.sha256(manifest[:-32]).digest()
    (directory / "manifest.etm").write_bytes(manifest)


def mutate_u32(path: Path, offset: int, value: int, *, rehash: bool = False) -> None:
    data = bytearray(path.read_bytes())
    struct.pack_into("<I", data, offset, value)
    if rehash:
        data[-32:] = hashlib.sha256(data[:-32]).digest()
    path.write_bytes(data)


def build(output: Path) -> None:
    output.mkdir(parents=True)
    small = output / "small"
    small.mkdir()
    write_d1_resource(small, ((1, 2),), fingerprint=FINGERPRINT, vocab=VOCAB)

    for name, shards in (
        ("empty", ()),
        ("singleton", ((1,),)),
        ("exact", ((1, 2, 3, 4),)),
        ("padded", ((1, 2, 3),)),
        ("cross-shard", ((10, 11), (12, 13))),
        ("shuffle-cross", tuple((value, value + 1) for value in range(0, 20, 2))),
    ):
        destination = output / name
        destination.mkdir()
        write_d1_resource(destination, shards, fingerprint=FINGERPRINT, vocab=VOCAB)

    # The native exact-read seam receives a UTF-8 byte count, not a Scheme
    # character count. Keep this fixture non-ASCII so the compiled path is
    # exercised on every focused run.
    clone(small, output / "café-corpus")

    large = output / "large"
    large.mkdir()
    shards = tuple(
        tuple((base + offset) % VOCAB for offset in range(min(256, 8193 - base)))
        for base in range(0, 8193, 256)
    )
    write_d1_resource(large, shards, fingerprint=FINGERPRINT, vocab=VOCAB)

    # Equal-token corpora whose only material difference is shard topology.
    # Equal-length leaf names keep path allocation from biasing the retained
    # arena comparison.
    arena_one = output / "one-shards"
    arena_one.mkdir()
    write_d1_resource(
        arena_one, (tuple(index % VOCAB for index in range(1024)),),
        fingerprint=FINGERPRINT, vocab=VOCAB,
    )
    arena_many = output / "many-shard"
    arena_many.mkdir()
    write_d1_resource(
        arena_many, tuple((index % VOCAB,) for index in range(1024)),
        fingerprint=FINGERPRINT, vocab=VOCAB,
    )
    if (arena_one / "manifest.etm").stat().st_size != 279:
        raise AssertionError("D2 one-shard arena manifest size drifted")
    if (arena_many / "manifest.etm").stat().st_size != 57_567:
        raise AssertionError("D2 many-shard arena manifest size drifted")
    if (arena_one / "shard-0000000000000000.ets").stat().st_size != 8_399:
        raise AssertionError("D2 one-shard arena fixture size drifted")
    if (arena_many / "shard-0000000000000000.ets").stat().st_size != 215:
        raise AssertionError("D2 many-shard arena fixture size drifted")

    missing_manifest = output / "missing-manifest"
    missing_manifest.mkdir()
    shutil.copy2(small / "shard-0000000000000000.ets", missing_manifest)

    missing_late = clone(large, output / "missing-late-shard")
    (missing_late / "shard-0000000000000032.ets").unlink()

    corrupt_manifest = clone(small, output / "corrupt-manifest-checksum")
    manifest_path = corrupt_manifest / "manifest.etm"
    data = bytearray(manifest_path.read_bytes())
    data[-33] ^= 1
    manifest_path.write_bytes(data)

    version_manifest = clone(small, output / "version-manifest")
    data = bytearray((version_manifest / "manifest.etm").read_bytes())
    struct.pack_into("<H", data, 8, 2)
    (version_manifest / "manifest.etm").write_bytes(data)

    feature_manifest = clone(small, output / "feature-manifest")
    mutate_u32(feature_manifest / "manifest.etm", 16, 1)

    corrupt_shard = clone(small, output / "corrupt-shard-checksum")
    shard_path = corrupt_shard / "shard-0000000000000000.ets"
    data = bytearray(shard_path.read_bytes())
    data[-33] ^= 1
    shard_path.write_bytes(data)

    version_shard = clone(small, output / "version-shard")
    shard_path = version_shard / "shard-0000000000000000.ets"
    data = bytearray(shard_path.read_bytes())
    struct.pack_into("<H", data, 8, 2)
    shard_path.write_bytes(data)

    feature_shard = clone(small, output / "feature-shard")
    mutate_u32(feature_shard / "shard-0000000000000000.ets", 16, 1)

    invalid_token = clone(small, output / "invalid-token")
    shard_path = invalid_token / "shard-0000000000000000.ets"
    data = bytearray(shard_path.read_bytes())
    header_bytes = struct.unpack_from("<I", data, 12)[0]
    struct.pack_into("<Q", data, header_bytes, VOCAB)
    shard_path.write_bytes(data)
    bind_shard(invalid_token, 0)

    singleton = output / "corrupt-singleton"
    singleton.mkdir()
    write_d1_resource(singleton, ((1,), (2,), (3,)), fingerprint=FINGERPRINT, vocab=VOCAB)
    shard_path = singleton / "shard-0000000000000001.ets"
    data = bytearray(shard_path.read_bytes())
    data[-33] ^= 1
    shard_path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
