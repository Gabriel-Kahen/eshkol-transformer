#!/usr/bin/env python3
"""Build a source-composed E3 variant missing failed-bind K1 cleanup."""

from pathlib import Path
import shutil
import sys


root = Path(__file__).resolve().parents[2]
output = Path(sys.argv[1])
source = (root / "src/eshkol_transformer/e3_frame.c").read_text()
original = """    for (size_t i = 6; i > 0; --i) {
      et_kernel_runtime_destroy(frame->runtime[i - 1]);
      frame->runtime[i - 1] = NULL;
    }
    e3_destroy_owned(frame);"""
if source.count(original) != 1:
    raise SystemExit("failed-bind runtime cleanup anchor changed")
mutant = original.replace(
    "      et_kernel_runtime_destroy(frame->runtime[i - 1]);\n", ""
)
destination = output / "src/eshkol_transformer/e3_frame.c"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(source.replace(original, mutant))
test_destination = output / "tests/e3_native/test_frame.c"
test_destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(root / "tests/e3_native/test_frame.c", test_destination)
