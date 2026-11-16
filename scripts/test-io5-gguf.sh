#!/usr/bin/env bash
set -euo pipefail
io5_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${io5_root}"
mkdir -p "${io5_root}/.tmp/scratch"
export TMPDIR="${io5_root}/.tmp/scratch"
if [[ -n "${IO5_GGUF_REFERENCE_PYTHON:-}" ]]; then
  "${IO5_GGUF_REFERENCE_PYTHON}" -c 'import importlib.metadata, gguf, numpy; assert importlib.metadata.version("gguf") == "0.19.0"; assert numpy.__version__ == "2.5.4"'
  export IO5_REFERENCE_PYTHON="${IO5_GGUF_REFERENCE_PYTHON}"
fi
bash scripts/test-io5.sh
