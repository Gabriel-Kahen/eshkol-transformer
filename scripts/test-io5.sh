#!/usr/bin/env bash
set -euo pipefail
io5_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${io5_root}"
if [[ -n "${IO5_REFERENCE_PYTHON:-}" ]]; then
  "${IO5_REFERENCE_PYTHON}" -c 'import safetensors; assert safetensors.__version__ == "0.6.2", "IO5 reference requires safetensors==0.6.2"'
fi
"${IO5_REFERENCE_PYTHON:-python3}" -m unittest discover -s tests/io5 -p 'test_*.py' -v
python3 -m unittest discover -s tests/q0 -p test_python_isolation.py -v
