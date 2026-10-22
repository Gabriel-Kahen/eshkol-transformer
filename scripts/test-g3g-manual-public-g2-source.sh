#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
bash -n scripts/build-e1b-consumer.sh scripts/g3g-manual-native-inputs.sh \
  scripts/test-g3g-manual-public-g2.sh
python3 -B tests/g3g_manual/check_public_g2_gate.py
python3 -B -m unittest -v tests.g3g_manual.test_public_g2_gate tests.q0.test_python_isolation
python3 -B scripts/check-g3g-manual-package.py
python3 -B tests/g3c4/test_public_g2_port_source.py
git diff --check
