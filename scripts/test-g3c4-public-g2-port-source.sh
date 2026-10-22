#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
bash scripts/test-g3c4-p2g2-terminal-commit-source.sh
bash scripts/test-g3c4-p2g2-terminal-ids-owner-source.sh
bash scripts/test-g3c4-p2g2-terminal-snapshot-source.sh
bash scripts/test-g3c4-p2g2-terminal-release-source.sh
bash scripts/test-g3c4-p2g2-typed-wrapper-source.sh
python3 -B scripts/check-g3g-manual-package.py
python3 -B tests/g3c4/test_public_g2_port_source.py
bash scripts/test-g3c4-public-g2-port-native-source.sh
