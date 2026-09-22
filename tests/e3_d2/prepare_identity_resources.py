#!/usr/bin/env python3
"""Deterministic development fixtures; production uses canonical T1/D1 readers."""
import argparse
from pathlib import Path

from tests.d2.test_resources import write_d1_resource
from tests.t1.reference import Identity, Special, encode_artifact, fingerprint


def build(root: Path) -> None:
    root.mkdir(parents=True)
    special = (Special(256, 'bos', 'omit'),)
    profiles = {
        'raw': Identity(),
        'strict': Identity(utf8_policy='strict'),
        'special': Identity(specials=special),
        'prefix': Identity(specials=special, prefix=(256,)),
        'suffix': Identity(specials=special, suffix=(256,)),
    }
    for name, profile in profiles.items():
        (root / f'{name}.tsv').write_bytes(encode_artifact(profile))
    for name, identity, vocab in (
        ('matching', fingerprint(Identity()), 256),
        ('wrong-fingerprint', fingerprint(Identity(utf8_policy='strict')), 256),
        ('wrong-vocab', fingerprint(Identity()), 257),
    ):
        destination = root / name
        destination.mkdir()
        write_d1_resource(destination, ((0, 65, 255),), fingerprint=identity, vocab=vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
