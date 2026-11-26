#!/usr/bin/env python3
"""Generate private test inputs for a C1/C2 whole-input SHA feasibility run."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native/c2_checkpoint_format.c"

ENTRY = r'''

/* Appended only to a generated developer copy of c2_checkpoint_format.c. */
int64_t et_c2_development_sha256_whole_v1(const void *input_header,
                                           int64_t length, void *output_header) {
  sha256_state state;
  int64_t observed_input, observed_output;
  const uintptr_t input = (uintptr_t)input_header;
  const uintptr_t output = (uintptr_t)output_header;
  if (input_header == NULL || output_header == NULL || length < 0 ||
      length > INT64_C(2305843009213693951) ||
      (uint64_t)length > (uint64_t)SIZE_MAX - sizeof(int64_t) ||
      input > UINTPTR_MAX - sizeof(int64_t) - (uint64_t)length ||
      output > UINTPTR_MAX - sizeof(int64_t) - 32u) return -1;
  memcpy(&observed_input, input_header, sizeof(observed_input));
  memcpy(&observed_output, output_header, sizeof(observed_output));
  if (observed_input != length || observed_output != 32) return -1;
  sha256_init(&state);
  sha256_update(&state, (const uint8_t *)input_header + sizeof(int64_t),
                (size_t)length);
  sha256_final(&state, (uint8_t *)output_header + sizeof(int64_t));
  return 0;
}
'''

OVERLAY = r'''
;;; Generated developer-only substitution. The accepted C1 module is unchanged.
(extern i64 c2-development-sha256-whole ptr i64 ptr
        :real et_c2_development_sha256_whole_v1)
(define c2-development-original-sha256 c1-sha256)
(define (c2-development-native-sha256 bytes)
  (if (not (bytevector? bytes))
      (raise "c1-sha256: expected a bytevector"))
  (if (> (bytevector-length bytes) 2305843009213693951)
      (raise "c1-sha256: input exceeds the SHA-256 length field"))
  (let* ((digest (make-bytevector 32 0))
         (status (c2-development-sha256-whole
                   bytes (bytevector-length bytes) digest)))
    (if (not (= status 0))
        (raise "development native SHA-256 failed"))
    digest))
(define (c2-development-parity-input size)
  (let ((bytes (make-bytevector size 0)))
    (let fill ((index 0))
      (if (< index size)
          (begin (bytevector-u8-set! bytes index
                   (modulo (+ (* index 17) 91) 256))
                 (fill (+ index 1)))))
    bytes))
(define (c2-development-sha-parity)
  (let check ((sizes '(0 1 3 55 56 57 63 64 65 119 120 127 128 129)))
    (if (null? sizes) #t
        (let ((bytes (c2-development-parity-input (car sizes))))
          (if (not (c1-bytes=? (c2-development-original-sha256 bytes)
                              (c2-development-native-sha256 bytes)))
              (raise "development SHA-256 parity failed"))
          (check (cdr sizes))))))
(c2-development-sha-parity)
'''


def substitute_c1(c1_overlay: Path) -> None:
    anchor = "(define (c2-timed-c1-hash-parts parts stage)"
    call = "(digest (c1-sha256 input))"
    source = c1_overlay.read_text()
    if (source.count(anchor) != 1 or source.count("(require c1_sha256)") != 1
            or source.count(call) != 1):
        raise ValueError("C1 overlay anchors changed")
    source = source.replace(anchor, OVERLAY + "\n" + anchor)
    c1_overlay.write_text(source.replace(call, "(digest (c2-development-native-sha256 input))"))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("native", "overlay"):
        raise SystemExit("usage: c2_native_sha_feasibility.py native GENERATED_C | overlay C1_OVERLAY")
    if sys.argv[1] == "native":
        Path(sys.argv[2]).write_text(NATIVE.read_text() + ENTRY)
    else:
        substitute_c1(Path(sys.argv[2]))
