#!/usr/bin/env python3
"""Generate the noninstalled zero tuple's one-shot real P1 wrong-tie control."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "native/m3t_transport_extension.esk"
CANONICAL_SHA = "71247e722d72764ce1b3400cb3f818ba12c3754dd11a6c742ffcec63942d5815"
VARIANT_SHA = "17dc36a45e56c94aa2bc584c4132ce052ba2349a00684cfd436fca21714eeae1"
ANCHOR = '''(let ((parent (m3t-child-parent root '("token_embedding" "weight") cache)))
              (i2-module-register-tied-parameter-internal!
                parent "weight" (vector-ref (vector-ref entry 8) 10)))'''
REPLACEMENT = '''(let ((parent (m3t-child-parent root '("token_embedding" "weight") cache)))
              (i2-module-register-tied-parameter-internal!
                parent "weight"
                (m3t-zero-test-tie-target (vector-ref entry 8))))'''
SEAM = ''';;; Noninstalled zero-control tuple only. Consume the armed fault before
;;; authentic P1 tied registration; the canonical retry uses handle 10.
(define m3t-zero-test-wrong-tie-cell (vector #f))
(define (m3t-zero-test-wrong-tie-arm!)
  (if (vector-ref m3t-zero-test-wrong-tie-cell 0)
      (m3t-fail 'invalid-state 'diagnostic-model-create
        "wrong-tie control is already armed"))
  (vector-set! m3t-zero-test-wrong-tie-cell 0 #t))
(define (m3t-zero-test-tie-target handles)
  (if (vector-ref m3t-zero-test-wrong-tie-cell 0)
      (begin
        (vector-set! m3t-zero-test-wrong-tie-cell 0 #f)
        (vector-ref handles 9))
      (vector-ref handles 10)))

'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compose(source: bytes) -> bytes:
    if sha(source) != CANONICAL_SHA:
        raise ValueError("canonical M3T source changed")
    text = source.decode("utf-8")
    if text.count(ANCHOR) != 1 or text.count("(define (m3t-public-model-create ") != 1:
        raise ValueError("M3T constructor anchors changed")
    text = text.replace(ANCHOR, REPLACEMENT, 1)
    text = text.replace("(define (m3t-public-model-create ",
                        SEAM + "(define (m3t-public-model-create ", 1)
    generated = text.encode("utf-8")
    if sha(generated) != VARIANT_SHA:
        raise ValueError("zero M3T variant digest differs")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = compose(CANONICAL.read_bytes())
    if args.check:
        if args.output.read_bytes() != expected:
            raise ValueError("generated zero M3T variant changed")
    else:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("generated zero M3T output already exists")
        args.output.write_bytes(expected)
    print(sha(expected))


if __name__ == "__main__":
    main()
