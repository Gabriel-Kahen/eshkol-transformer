#!/usr/bin/env python3
"""Reconstruct the immutable EOS predecessor and check second-frame boundaries."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys

MACRO = 'ET_G3C4_P2_G2_SECOND_FRAME_PRIVATE'

def strip_second_frame(text):
    lines = text.splitlines(keepends=True)
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r'^#(?:if|ifdef)\b', line) and MACRO in line:
            j = i
            while lines[j].rstrip().endswith('\\'):
                j += 1
            depth, alternate = 1, None
            body = j + 1
            j += 1
            while j < len(lines) and depth:
                if re.match(r'^#(?:if|ifdef|ifndef)\b', lines[j]):
                    depth += 1
                elif re.match(r'^#endif\b', lines[j]):
                    depth -= 1
                elif depth == 1 and re.match(r'^#else\b', lines[j]):
                    alternate = j + 1
                elif depth == 1 and re.match(r'^#elif\b', lines[j]):
                    raise ValueError('unsupported second-frame inverse branch')
                if depth:
                    j += 1
            if depth:
                raise ValueError('unterminated second-frame feature block')
            if alternate is not None:
                result.extend(lines[alternate:j])
            i = j + 1
        else:
            result.append(line)
            i += 1
    text = ''.join(result)
    # Exact insertion spacing, verified by immutable predecessor hashes below.
    text = text.replace('#endif\n\n\n#include <limits.h>', '#endif\n\n#include <limits.h>')
    text = text.replace('#endif\n\n\nint64_t et_g3c4_private_output_copy_decode_ids_v1(',
                        '#endif\n\nint64_t et_g3c4_private_output_copy_decode_ids_v1(')
    text = text.replace('sizeof(token_candidate));\n  return 0;\n}\n\n#endif\n#endif\n#endif',
                        'sizeof(token_candidate));\n  return 0;\n}\n#endif\n#endif\n#endif')
    text = text.replace('#endif\n\n\n#ifdef __cplusplus', '#endif\n\n#ifdef __cplusplus')
    return text

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', type=Path)
    args = parser.parse_args()
    for name, digest in (
        ('g3c4_model_owner.c', 'b1741b95ce4f75dc27bcdd38e9ec5332c768653f2ee8b085bc8444ce0469c477'),
        ('g3c4_context_internal.h', '0a3c2a7873f90e3ceb2612dcf40f11db6ceaa03d6c01af49752d940cf9866416'),
    ):
        source = (root / 'src/eshkol_transformer' / name).read_text()
        baseline = strip_second_frame(source)
        if hashlib.sha256(baseline.encode()).hexdigest() != digest:
            if args.baseline_dir:
                args.baseline_dir.mkdir(parents=True, exist_ok=True)
                (args.baseline_dir / name).write_text(baseline)
            raise SystemExit(f'second-frame predecessor byte mismatch: {name}')
        if args.baseline_dir:
            args.baseline_dir.mkdir(parents=True, exist_ok=True)
            (args.baseline_dir / name).write_text(baseline)
    subprocess.run([sys.executable, str(root/'scripts/check-g3c4-p2g2-eos-first-frame.py')],
                   cwd=root, check=True, stdout=subprocess.DEVNULL)
    owner = (root/'src/eshkol_transformer/g3c4_model_owner.c').read_text()
    test = (root/'tests/g3c4/test_p2_g2_second_frame.c').read_text()
    for value in ('et_g3c4_p2g2_second_cache_preflight(context)',
                  'context, last_logits, &selected, 1)',
                  'context->p2g2_second_frame == 1u',
                  'et_g3c4_p2g2_second_output_clean(context, output)',
                  'candidate, last_logits, speculative_token_output, 0)',
                  'memcmp(last_logits, context->p2g2_prefix_next_logits',
                  'C4 P2/G2 second frame requires the authenticated prefix and carrier bridge'):
        if value not in owner:
            raise SystemExit(f'second-frame source invariant missing: {value}')
    for value in ('SECOND-ORACLE-PAIR', 'capture_t4 = 1', 'reference_t4.keys',
                  'pending_ids_zero(output)', 'check_cache_snapshot_equal(&prefix, &after)',
                  'fake_id == -17', 'fail_bridge_sample = cut == 0',
                  'forged_token = cut == 2 ? -1 : 256', 'second_tamper = &carrier',
                  'et_a2_kv_cache_transaction_view_end_v1',
                  'et_g3c4_private_call_abort_v1(context)'):
        if value not in test:
            raise SystemExit(f'second-frame test invariant missing: {value}')
    print('G3-C4 P2/G2 second-frame static PASS; immutable predecessor bytes preserved')

if __name__ == '__main__':
    main()
