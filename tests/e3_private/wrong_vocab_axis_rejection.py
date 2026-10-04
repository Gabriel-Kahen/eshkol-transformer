"""Classify one exact private compiled wrong-axis rejection without hiding diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import traceback

SOURCE = Path(__file__).resolve().parents[2]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from tests.e3_reference.transcript import TranscriptError


ERROR = "batches[0].correct_f32: target/argmax wiring mismatch"
MARKER = f"E3-WRONG-VOCAB-AXIS-EXPECTED: {ERROR}\n".encode("ascii")
NUMPY_WARNING = (
    b"/home/gabe/.codex/venvs/eshkol-oracle-py3146/lib/python3.14/"
    b"site-packages/torch/_subclasses/functional_tensor.py:368: UserWarning: "
    b"Failed to initialize NumPy: No module named 'numpy' (Triggered internally "
    b"at /__w/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)\n"
    b"  cpu = _conversion_method_template(device=torch.device(\"cpu\"))\n"
)
STATUS = re.compile(rb"(?:0|[1-9][0-9]{0,2})\n\Z")
TRANSCRIPT = SOURCE / "tests/e3_reference/transcript.py"
VERIFIER = SOURCE / "tests/e3_native_parity/verify_observation.py"


def _regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or nonregular rejection evidence: {path.name}")
    return path.read_bytes()


def _expected_error(error: Exception) -> bool:
    if type(error) is not TranscriptError or str(error) != ERROR:
        return False
    frames = traceback.extract_tb(error.__traceback__)
    return (len(frames) >= 3
            and frames[-2].filename == str(TRANSCRIPT)
            and frames[-2].name == "validate_observed_case"
            and frames[-1].filename == str(TRANSCRIPT)
            and frames[-1].name == "_fail"
            and any(frame.filename == str(VERIFIER) and frame.name == "main"
                    for frame in frames))


def run(arguments: argparse.Namespace) -> int:
    from tests.e3_native_parity.verify_observation import main as verify_main

    if arguments.traceback.exists() or arguments.transcript.exists():
        raise ValueError("wrong-axis rejection output must be fresh")
    original_argv = sys.argv
    sys.argv = [str(VERIFIER), "--reference", str(arguments.reference),
                "--case", "all-correct", "--observed", str(arguments.observed),
                "--roles", str(arguments.roles), "--transcript", str(arguments.transcript)]
    try:
        try:
            verify_main()
        except Exception as error:
            with arguments.traceback.open("xb") as output:
                output.write("".join(traceback.format_exception(error)).encode("utf-8"))
            if not _expected_error(error):
                raise
            sys.stderr.buffer.write(MARKER)
            return 1
    finally:
        sys.argv = original_argv
    return 0


def check(arguments: argparse.Namespace) -> None:
    status = _regular(arguments.status)
    if not STATUS.fullmatch(status) or int(status) != 1:
        raise ValueError("wrong-axis validator exit must be canonical numeric 1")
    if _regular(arguments.stdout):
        raise ValueError("wrong-axis validator stdout must be empty")
    stderr = _regular(arguments.stderr)
    if stderr not in (MARKER, NUMPY_WARNING + MARKER):
        raise ValueError("wrong-axis validator diagnostic differs from exact expected output")
    trace = _regular(arguments.traceback)
    expected_tail = ("tests.e3_reference.transcript.TranscriptError: "
                     + ERROR + "\n").encode("ascii")
    if (not trace.startswith(b"Traceback (most recent call last):\n")
            or not trace.endswith(expected_tail)
            or trace.count(b"Traceback (most recent call last):") != 1
            or str(VERIFIER).encode() not in trace
            or str(TRANSCRIPT).encode() not in trace
            or b"in validate_observed_case\n" not in trace
            or b"in _fail\n" not in trace):
        raise ValueError("wrong-axis raw validator traceback lacks exact source context")
    if arguments.transcript.exists() or arguments.transcript.is_symlink():
        raise ValueError("rejected wrong-axis transcript was published")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    for flag in ("reference", "observed", "roles", "transcript", "traceback"):
        execute.add_argument(f"--{flag}", required=True, type=Path)
    audit = sub.add_parser("check")
    for flag in ("status", "stdout", "stderr", "traceback", "transcript"):
        audit.add_argument(f"--{flag}", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.command == "run":
        return run(arguments)
    check(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
