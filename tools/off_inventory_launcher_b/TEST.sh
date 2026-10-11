#!/bin/sh
set -eu
cd "$(dirname "$0")"
/usr/bin/python3 -B -m unittest -v TEST_RUNNER.py
/usr/bin/python3 - <<'PY'
import subprocess
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory(prefix='eshkol-off-launcher-b-test-') as temp:
    binary = Path(temp) / 'TEST_LAUNCHER.test'
    subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-static', '-o', str(binary), 'TEST_LAUNCHER.c'], check=True)
    subprocess.run([str(binary)], check=True)
    launcher = Path(temp) / 'LAUNCHER.test'
    subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-static', '-o', str(launcher), 'LAUNCHER.c'], check=True)
    result = subprocess.run([str(launcher), '/tmp/synthetic-output'],
                            text=True, capture_output=True)
    assert result.returncode == 2 and 'setuid-root launch contract absent' in result.stderr
    print('unprivileged launcher: expected HOLD (2)')
PY
