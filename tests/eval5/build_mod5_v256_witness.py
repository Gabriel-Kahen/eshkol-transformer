"""Pin and specialize the accepted MOD5-C test source without copying its model code."""

import hashlib
from pathlib import Path
import sys

from tests.eval5.mod5_v256_fixture import verify

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / "tests/mod5/mod5_e2e_c1_io5.esk"
TAIL = Path(__file__).with_name("mod5_v256_witness_tail.esk")
OWNER_SHA256 = "593d9de4dc03521359941bacece2d159c7e022f96f4b0752943b83fc6ae3c31c"


def one(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"MOD5 source contract drift: expected one {before!r}")
    return text.replace(before, after)


def digest_literal(hex_digest):
    return "(bytevector " + " ".join(str(value) for value in bytes.fromhex(hex_digest)) + ")"


def generate(fixture_root):
    record = verify(fixture_root)
    owner_bytes = OWNER.read_bytes()
    if hashlib.sha256(owner_bytes).hexdigest() != OWNER_SHA256:
        raise ValueError("MOD5-C owner source changed; review the extraction before use")
    owner = owner_bytes.decode("utf-8")
    if owner.count("(define (initial-weights tied?)") != 1 or owner.count("(define hex-digits") != 1:
        raise ValueError("MOD5-C extraction boundaries changed")
    prefix = owner.split("(define (initial-weights tied?)", 1)[0]
    checkpoint = owner.split("(define (buffer->bytes buffer count)", 1)[1]
    checkpoint = "(define (buffer->bytes buffer count)" + checkpoint.split("(define hex-digits", 1)[0]
    # The existing forward path, K1 bridge, and C1/P1/I2 loader remain verbatim.
    source = prefix + checkpoint
    source = one(source, "(define vocab 16)", "(define vocab 256)")
    source = one(source, '("head" "weight") (16 16)', '("head" "weight") (256 16)')
    source = one(source, '("token_embedding" "weight") (16 16)',
                 '("token_embedding" "weight") (256 16)')
    source = one(source, "(define span-bytes (* 4 width hidden))",
                 "(define span-bytes (* 4 vocab width))")
    source = one(source, '(define corpus-identity "mod5-e2e-token-v1")',
                 f'(define corpus-identity "{record["tokenizer_fingerprint"]}")')
    # Only the formatting helper is needed from the removed training section.
    formatter = owner.split("(define (print-real format value)", 1)[1]
    formatter = "(define (print-real format value)" + formatter.split("(define (train input target tied?)", 1)[0]
    tail = TAIL.read_text()
    tail = one(tail, "@@MODEL_SHA@@", digest_literal(record["checkpoint_sha256"]))
    tail = one(tail, "@@TRAIN_SHA@@",
               digest_literal(record["splits"]["train"]["dataset_sha256"]))
    tail = one(tail, "@@EVAL_SHA@@",
               digest_literal(record["splits"]["eval-clean"]["dataset_sha256"]))
    return source + formatter + tail


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: python3 -m tests.eval5.build_mod5_v256_witness FIXTURE OUTPUT.esk")
    output = Path(argv[2])
    if output.exists() or output.is_symlink():
        raise FileExistsError("witness source destination must be fresh")
    output.write_text(generate(Path(argv[1])))


if __name__ == "__main__":
    main(sys.argv)
