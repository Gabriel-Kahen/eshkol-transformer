#!/usr/bin/env python3
"""Prove the clone-disabled C4 surface matches the accepted terminal base."""
import argparse
from pathlib import Path
import runpy
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = {
    'src/eshkol_transformer/g3c4_model_owner.c':
        '1cafa1be6b798af11e738f9f745110a800f2ca2d',
    'src/eshkol_transformer/g3c4_context_internal.h':
        '108e54e7e77d17228bba3610eeb31f06d789fe94',
}


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
    # The accepted terminal checker owns this established private macro set.
    namespace = runpy.run_path(str(ROOT / 'scripts/check-g3c4-p2g2-terminal.py'))
    owner_macros = namespace['OWNER_MACROS'] + [
        'ET_G3C4_P2_G2_TERMINAL_PRIVATE',
        'ET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE',
    ]
    with tempfile.TemporaryDirectory(prefix='g3c4-terminal-ids-off-') as folder:
        baseline = Path(folder)
        for name, blob in BASE.items():
            old = baseline / name
            old.parent.mkdir(parents=True, exist_ok=True)
            old.write_bytes(subprocess.run(
                ['git', 'cat-file', 'blob', blob], check=True, cwd=ROOT,
                capture_output=True).stdout)
            macros = owner_macros if name.endswith('.c') else [
                'ET_G3C4_P2_G2_TERMINAL_PRIVATE']
            if preprocess(args.cc, old, macros) != preprocess(
                    args.cc, ROOT / name, macros):
                raise SystemExit(f'clone-off preprocessor drift: {name}')
    print('G3-C4 P2/G2 logical I1 clone-off preprocessing PASS')


if __name__ == '__main__':
    main()
