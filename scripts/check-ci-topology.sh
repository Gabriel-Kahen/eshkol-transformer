#!/usr/bin/bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "${project_root}/tests/ci/topology.py"
