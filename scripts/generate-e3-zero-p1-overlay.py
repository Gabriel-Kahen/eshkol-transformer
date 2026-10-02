#!/usr/bin/env python3
"""Emit the one exact, private P1 projection selector for the zero tuple."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "internal/p1/lib/transformer/module.esk"
SOURCE_SHA256 = "1146eee9e053b4a761c24f56b75ed43e756ec2d907d0de48b04e72abe805775d"
ANCHOR = """  (lambda (identity)
    (construction-seal-prepared! identity))
  )))"""
SLOT = """  (lambda (identity)
    (construction-seal-prepared! identity))
  ;; Zero-control-only slot 73. No callback or user entry in the mutation span.
  (lambda (model modules handles carriers selector)
    (if (not (or (eq? selector 'path-10)
                 (eq? selector 'shape-10)
                 (eq? selector 'tie-10))) 1
        (let ((baseline (tr3-fixed-recheck! model modules handles carriers)))
          (if (not (= baseline 0)) baseline
              (let* ((head (tr3-fixed-shell-raw (vector-ref handles 10) 'handle))
                     (other (tr3-fixed-shell-raw (vector-ref handles 13) 'handle))
                     (token (tr3-fixed-module-raw modules 16))
                     (bindings (if token (vector-ref token 2) '()))
                     (binding (if (pair? bindings) (car bindings) #f))
                     (wrong-path '("head" "wrong"))
                     (wrong-shape '(255 4)))
                (if (not (and (handle-value? head) (handle-value? other)
                              (vector? binding) (= (vector-length binding) 3)
                              (eq? (vector-ref binding 0) 'parameter-binding)
                              (string=? (vector-ref binding 1) "weight")
                              (eq? (vector-ref binding 2) head)
                              (tr3-fixed-path? (vector-ref head 2) 10)
                              (tr3-fixed-shape? (vector-ref head 3) 10))) 4
                    (let* ((target (if (eq? selector 'tie-10) binding head))
                           (field (if (eq? selector 'path-10) 2
                                      (if (eq? selector 'shape-10) 3 2)))
                           (original (vector-ref target field))
                           (replacement
                             (cond ((eq? selector 'path-10) wrong-path)
                                   ((eq? selector 'shape-10) wrong-shape)
                                   (else other)))
                           (status #f))
                      (guard (caught
                               (#t (vector-set! target field original)
                                   (if (not (= (tr3-fixed-recheck!
                                                 model modules handles carriers) 0))
                                       (fail 'internal 'e3-zero-projection-test
                                             "P1 projection restoration failed" '()))
                                   (raise caught)))
                        (vector-set! target field replacement)
                        (set! status
                          (tr3-fixed-recheck! model modules handles carriers))
                        (vector-set! target field original))
                      (if (not (= (tr3-fixed-recheck!
                                    model modules handles carriers) 0))
                          (fail 'internal 'e3-zero-projection-test
                                "P1 projection restoration failed" '()))
                      (if (not (= status 4))
                          (fail 'internal 'e3-zero-projection-test
                                "P1 projection mismatch was not rejected" '()))
                      status))))))
  ))))"""


def rendered() -> bytes:
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("canonical P1 module bytes differ from the reviewed predecessor")
    text = source.decode("utf-8")
    if text.count(ANCHOR) != 1:
        raise ValueError("P1 slot-73 insertion anchor is not unique")
    return text.replace(ANCHOR, SLOT).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output_dir / "p1/transformer/module.esk"
    expected = rendered()
    if args.check:
        if output.read_bytes() != expected:
            raise ValueError("generated private P1 overlay differs")
    else:
        if output.exists() or output.is_symlink():
            raise ValueError("private P1 overlay path must be fresh")
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        output.write_bytes(expected)
    print(hashlib.sha256(expected).hexdigest())


if __name__ == "__main__":
    main()
