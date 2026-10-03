#!/usr/bin/env python3
"""Compare terminal-off preprocessing with the immutable second-frame tree."""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE_BLOBS = {
    'src/eshkol_transformer/g3c4_model_owner.c':
        '5d75302878aaedc1c3e134e91038ad21c2a91489',
    'src/eshkol_transformer/g3c4_context_internal.h':
        'e39caff71826d1ce7af9ac1cfbfd60e5444aa2ba',
    'include/eshkol_transformer/a2_kv_cache.h':
        'cc18e76bdde2c89ec86f23a37e7ad5ebf465c0bf',
    'native/a2_kv_cache.c':
        '2c502944181752f804d50b25c53be17784d81080',
}
OWNER_MACROS = [
    'ET_G3C4_CONTEXT_PRIVATE', 'ET_G3C4_NATIVE_PINS_PRIVATE',
    'ET_G3C4_ACTIVE_CALL_PRIVATE', 'ET_G3C4_GENERATOR_PRIVATE',
    'ET_G3C4_PROMPT_T1_BORROW_PRIVATE', 'ET_G3C4_PROVIDER_ROUTES_PRIVATE',
    'ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE',
    'ET_G3C4_SAMPLER_TRANSPORT_PRIVATE', 'ET_G3C4_TOKEN_FRAME_PRIVATE',
    'ET_G3C4_TOKEN_FORWARD_PRIVATE', 'ET_G3C4_PREFILL3_PRIVATE',
    'ET_G3C4_PREFILL1_PRIVATE', 'ET_G3C4_PREFILL2_PRIVATE',
    'ET_G3C4_PROMPT_PREFILL_PRIVATE', 'ET_G3C4_OUTPUT_RESERVATION_PRIVATE',
    'ET_G3C4_LAST_LOGIT_FRAME_PRIVATE', 'ET_G3C4_OUTPUT_PREPARE_PRIVATE',
    'ET_G3C4_OUTPUT_DECODE_IDS_PRIVATE', 'ET_G3C4_OUTPUT_TEXT_PRIVATE',
    'ET_G3C4_P2_G1_PENDING_PRIVATE', 'ET_G3C4_P2_G2_FIRST_FRAME_PRIVATE',
    'ET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE',
    'ET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE',
    'ET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE',
    'ET_G3C4_P2_G2_SECOND_FRAME_PRIVATE',
    'ET_I64_TENSOR_STORAGE_QUERY_PRIVATE',
    'ET_F32_TENSOR_STORAGE_QUERY_PRIVATE',
    'ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE',
]

def preprocess(cc, path, macros):
    command = [cc, '-std=c11', '-E', '-P', '-x', 'c',
               '-I', str(ROOT / 'include'), '-I', str(ROOT / 'native'),
               '-I', str(ROOT / 'src'),
               '-I', str(ROOT / 'src/eshkol_transformer'),
               *('-D' + macro for macro in macros), str(path)]
    return subprocess.run(command, check=True, cwd=ROOT,
                          capture_output=True).stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cc', required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='g3c4-terminal-off-') as folder:
        baseline = Path(folder)
        for name, blob in BASE_BLOBS.items():
            old = baseline / name
            old.parent.mkdir(parents=True, exist_ok=True)
            old.write_bytes(subprocess.run(
                ['git', 'cat-file', 'blob', blob], check=True, cwd=ROOT,
                capture_output=True).stdout)
            macros = OWNER_MACROS if name.endswith('g3c4_model_owner.c') else []
            before = preprocess(args.cc, old, macros)
            after = preprocess(args.cc, ROOT / name, macros)
            if before != after:
                raise SystemExit(f'terminal-off preprocessor drift: {name}')
    print('G3-C4 P2/G2 terminal-off four-source preprocessing PASS')

if __name__ == '__main__':
    main()
