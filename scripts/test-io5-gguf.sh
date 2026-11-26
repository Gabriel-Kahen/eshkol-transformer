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
export CC="${IO5_GGUF_NATIVE_CC:-clang}"
export CXX="${IO5_GGUF_NATIVE_CXX:-clang++}"
bash scripts/build-io5-quant.sh
export IO5_QUANT_REQUIRED=1
export ESHKOL_IO5_QUANT_LIB="${io5_root}/build/io5_quant/libeshkol_transformer_io5_quant.so"
bash scripts/test-io5.sh
if [[ "${IO5_GGUF_NATIVE_REQUIRED:-0}" == 1 || -n "${IO5_GGUF_NATIVE_PREFIX:-}${IO5_GGML_SOURCE_ARCHIVE:-}" ]]; then
  : "${IO5_GGUF_NATIVE_PREFIX:?set IO5_GGUF_NATIVE_PREFIX for the pinned GGML witness}"
  : "${IO5_GGML_SOURCE_ARCHIVE:?set IO5_GGML_SOURCE_ARCHIVE for the pinned GGML witness}"
  bash scripts/test-io5-q8-ggml-decoder.sh
else
  printf 'IO5 Q8 GGML decoder witness: SKIP (pinned native witness not configured)\n'
fi
