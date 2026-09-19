#!/usr/bin/env python3
import argparse
from pathlib import Path

from tests.d2.prepare_public_resources import build
from tests.d2.test_resources import write_d1_resource

BYTE_FINGERPRINT = (
    "sha256:eshkol-byte-tokenizer-v1:"
    "f4c24c4680961301963fb01723297c7a841bfa54241729d24e689401750221b8"
)

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
build(args.output)
byte = args.output / "byte-exact"
byte.mkdir()
write_d1_resource(byte, ((1, 2, 3, 4),),
                  fingerprint=BYTE_FINGERPRINT, vocab=259)
