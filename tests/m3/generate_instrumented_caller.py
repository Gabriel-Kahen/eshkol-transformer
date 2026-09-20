"""Add private observation only to a build-local copy of the public M3 caller."""
import argparse
from pathlib import Path


OBSERVATION = """(extern i64 m3-test-gradient-copy i64 ptr i64 :real et_m3_test_last_gradient_copy_v1)
(extern i64 m3-test-parameter-copy i64 ptr i64 :real et_m3_test_last_parameter_copy_v1)
(extern i64 m3-test-gradient-metadata i64 i64 :real et_m3_test_last_gradient_metadata_v1)
(define (m3-test-observe case-index)
  (let loop ((index 0) (sizes '(64 64 64 64 128 128 16 16 16 16 4096 16 16 32)))
    (if (null? sizes) #t
        (let ((gradient (make-bytevector (car sizes) 0))
              (parameter (make-bytevector (car sizes) 0)))
          (m3-test-check "gradient observation" (= (m3-test-gradient-copy index gradient (car sizes)) 0))
          (m3-test-check "parameter observation" (= (m3-test-parameter-copy index parameter (car sizes)) 0))
          (m3-test-emit-bytes (string-append "M3-GRADIENT " (number->string case-index)) index gradient)
          (m3-test-emit-bytes (string-append "M3-PARAMETER " (number->string case-index)) index parameter)
          (display "M3-METADATA ") (display case-index) (display " ") (display index)
          (display " ") (display (m3-test-gradient-metadata index 0))
          (display " ") (display (m3-test-gradient-metadata index 1))
          (display " ") (display (m3-test-gradient-metadata index 2)) (newline)
          (loop (+ index 1) (cdr sizes))))))
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).with_name("public_runtime.esk").read_text()
    begin, end = ";; M3-OBSERVATION-BEGIN\n", ";; M3-OBSERVATION-END\n"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError("public caller observation boundary changed")
    prefix, rest = source.split(begin)
    _, suffix = rest.split(end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prefix + OBSERVATION + suffix)
