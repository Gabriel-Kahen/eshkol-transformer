"""Admit only the pinned Q0 warning and the expected M2 comparator traceback."""

from __future__ import annotations

import argparse
from pathlib import Path


NUMPY_WARNING = (
    b"/q0/lib/python3.14/site-packages/torch/_subclasses/functional_tensor.py:368: "
    b"UserWarning: Failed to initialize NumPy: No module named 'numpy' "
    b"(Triggered internally at /__w/pytorch/pytorch/torch/csrc/utils/"
    b"tensor_numpy.cpp:84.)\n"
    b"  cpu = _conversion_method_template(device=torch.device(\"cpu\"))\n"
)
ERROR = (b"tests.e3_reference.transcript.TranscriptError: "
         b"batches[2].reduction_f32: exact rational reduction mismatch\n")
MUTANT_TRACEBACK = (
    b"Traceback (most recent call last):\n"
    b'  File "<frozen runpy>", line 203, in _run_module_as_main\n'
    b'  File "<frozen runpy>", line 88, in _run_code\n'
    b'  File "/source/tests/e3_native_parity/verify_observation.py", line 120, in <module>\n'
    b"    main()\n"
    b"    ~~~~^^\n"
    b'  File "/source/tests/e3_native_parity/verify_observation.py", line 72, in main\n'
    b"    validate_observed_case(observed, expected_batches)\n"
    b"    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
    b'  File "/source/tests/e3_reference/transcript.py", line 317, in validate_observed_case\n'
    b'    _fail(f"{where}.reduction_f32: exact rational reduction mismatch")\n'
    b"    ~~~~~" + b"^" * 61 + b"\n" +
    b'  File "/source/tests/e3_reference/transcript.py", line 32, in _fail\n'
    b"    raise TranscriptError(message)\n"
    + ERROR
)


def check(raw: bytes, *, mutant: bool) -> None:
    if raw.startswith(NUMPY_WARNING):
        raw = raw[len(NUMPY_WARNING):]
    if not mutant:
        if raw:
            raise ValueError("normal verifier emitted unrecognized stderr")
        return
    if raw != MUTANT_TRACEBACK:
        raise ValueError("mutant verifier stderr differs from expected rational rejection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("normal", "mutant"))
    parser.add_argument("stderr", type=Path)
    args = parser.parse_args()
    if args.stderr.is_symlink() or not args.stderr.is_file():
        raise ValueError("verifier stderr must be a regular file")
    check(args.stderr.read_bytes(), mutant=args.mode == "mutant")


if __name__ == "__main__":
    main()
