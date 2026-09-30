#!/usr/bin/env python3
"""Static boundary for the private P2/G2 prefix commit."""

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(root / "scripts/check-g3c4-p2g2-first-frame.py")],
               cwd=root, check=True, stdout=subprocess.DEVNULL)
owner = (root / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
test = (root / "tests/g3c4/test_p2_g2_prefix_commit.c").read_text()
runner = (root / "scripts/test-g3c4-p2g2-prefix-commit.sh").read_text()
for required in (
    "C4 P2/G2 prefix commit requires first frame and output ownership",
    "et_g3c4_private_p2g2_prefix_commit_v1(",
    "context->p2g2_prefix_committed = 1u;",
    "context->p2g2_prefix_next_logits",
    "context->p2g2_prefix_raw",
    "et_a2_kv_cache_transaction_commit_v1(",
    "et_g3c4_p2g2_prefix_clear(context);",
    "if (context->p2g2_prefix_committed != 0u)",
):
    if required not in owner:
        raise SystemExit(f"P2/G2 prefix source contract missing: {required}")
for required in (
    "prefix_case(owner, (const int64_t[2]){0, 255}, 0)",
    "prefix_case(owner, (const int64_t[2]){7, 11}, 1)",
    "check_cache_snapshot_equal(&committed, &reference_cache)",
    "et_g3c4_private_call_finish_v1(context)",
    "et_g3c4_private_call_abort_v1(context)",
    "et_i64_tensor_borrow_begin_v1(output->ids",
    "et_a2_kv_cache_read_borrow_begin_v1(",
    "fail_prefix_commit = 1",
    "WTERMSIG(status) == SIGABRT",
):
    if required not in test:
        raise SystemExit(f"P2/G2 prefix test contract missing: {required}")
for required in (
    "comm -3 \"$tmp/off.symbols\" \"$tmp/on.symbols\"",
    "printf '\\tet_g3c4_private_p2g2_prefix_commit_v1\\n'",
    "isolated P2/G2 prefix macro compiled",
    "-Wl,--wrap=et_a2_kv_cache_transaction_commit_v1",
    "check_p2_g1_oracle.py",
):
    if required not in runner:
        raise SystemExit(f"P2/G2 prefix runner contract missing: {required}")
print("G3-C4 P2/G2 prefix static PASS")
