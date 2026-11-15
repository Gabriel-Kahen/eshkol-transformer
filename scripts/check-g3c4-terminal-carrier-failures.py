#!/usr/bin/env python3
"""Classify null-return allocation cuts; they are not recoverable OOM proof."""
import os
from pathlib import Path
import resource
import subprocess
import sys

binary, mode, directory = sys.argv[1:]
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
env = dict(os.environ, ESHKOL_ARENA_POISON="1")
if mode == "sanitize":
    env.update(ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
               UBSAN_OPTIONS="halt_on_error=1")
for index in range(8):
    result = subprocess.run([binary, str(index)], env=env,
                            capture_output=True, timeout=120)
    prefix = Path(directory) / f"{mode}-allocation-{index}"
    prefix.with_suffix(".stdout").write_bytes(result.stdout)
    prefix.with_suffix(".stderr").write_bytes(result.stderr)
    marker = f"G3-C4 terminal carrier injected allocation failure: {index}\n".encode()
    assert result.stderr.startswith(marker), (index, result.returncode, result.stderr)
    assert result.returncode == 1, (index, result.returncode)
    assert result.stderr == marker + (
        b"Unhandled exception: object or exception-handler allocation failed\n")
    assert not result.stdout
    print(f"allocation-cut={index} mode={mode} null-return=unhandled-emergency status={result.returncode}")
