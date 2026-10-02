#!/usr/bin/env python3
"""Check the private EOS bridge boundary and reconstruct its reviewed predecessor."""

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--baseline-dir", type=Path)
args = parser.parse_args()
subprocess.run([sys.executable, str(root / "scripts/check-g3c4-p2g2-carrier-bridge.py")],
               cwd=root, check=True, stdout=subprocess.DEVNULL)
macro = "ET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE"
owner = (root / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
header = (root / "src/eshkol_transformer/g3c4_context_internal.h").read_text()


def replace_once(pattern, replacement, text):
    result, count = re.subn(pattern, replacement, text, flags=re.DOTALL)
    if count != 1:
        raise SystemExit(f"EOS bridge unique source anchor mismatch: {pattern}")
    return result


baseline_owner = replace_once(
    r"#if defined\(" + macro + r"\) && \\\n.*?#endif\n", "", owner)
baseline_owner = replace_once(
    r"#ifdef " + macro + r"\nstatic int64_t et_g3c4_p2g2_first_frame_carrier\(.*?"
    r"#else\n(.*?)#endif\n", r"\1", baseline_owner)
baseline_owner = replace_once(
    r"#ifdef " + macro + r"\n  if \(!allow_eos.*?#else\n(.*?)#endif\n",
    r"\1", baseline_owner)
baseline_owner = replace_once(
    r"#ifdef " + macro + r"\nint64_t et_g3c4_private_p2g2_first_frame_carrier_v1\("
    r".*?#endif\n", "", baseline_owner)
baseline_header = replace_once(
    r"#ifdef " + macro + r"\n.*?#endif\n\n", "", header)
for name, content, digest in (
    ("g3c4_model_owner.c", baseline_owner,
     "ba5fb3ec706a7f45c925b514dacdfb82181b9f1168cdf57267683f6c02b82f71"),
    ("g3c4_context_internal.h", baseline_header,
     "918d4e664bfed6069e8108c59f64e1d31f3eb27e80d94a97b2dee913b17a9854"),
):
    if hashlib.sha256(content.encode()).hexdigest() != digest:
        raise SystemExit(f"EOS bridge changed reviewed predecessor bytes: {name}")
    if args.baseline_dir:
        args.baseline_dir.mkdir(parents=True, exist_ok=True)
        (args.baseline_dir / name).write_text(content)
for required in (
    "C4 P2/G2 EOS first frame requires the authenticated carrier bridge",
    "if (!allow_eos && selected == context->generator_policy[5])",
    "context, input, staging_header, carrier_bytes, 0)",
    "context, input, staging_header, carrier_bytes, 1)",
):
    if required not in owner:
        raise SystemExit(f"EOS bridge source contract missing: {required}")
test = (root / "tests/g3c4/test_p2_g2_eos_first_frame.c").read_text()
for required in (
    "selected_eos ? selected : -1", "ET_G3C4_TOKEN_FRAME_READY",
    "check_cache_snapshot_equal(&after_abort, &prefilled)",
    "memcmp(actual_next, reference_next", "memcmp(actual_keys, full_prefix.keys",
    "memcmp(actual_values, full_prefix.values", "assert_pending(output)",
    "fail_bridge_sample = cut == 0", "fail_bridge_forward = cut == 1",
    "forged_token = 256", "carrier.canary[i] == 0x5au",
    "tampered_carrier->length = 9", "EOS-ORACLE-PAIR",
    "et_a2_kv_cache_transaction_view_begin_v1(",
):
    if required not in test:
        raise SystemExit(f"EOS bridge test contract missing: {required}")
runner = (root / "scripts/test-g3c4-p2g2-eos-first-frame.sh").read_text()
for required in ("off.preprocessed", "baseline.preprocessed", "symbols.diff",
                 "check_p2_g1_oracle.py", "test-g3c4-p2g2-carrier-bridge.sh",
                 "-fsanitize=address,undefined", "repeat.stdout"):
    if required not in runner:
        raise SystemExit(f"EOS bridge runner contract missing: {required}")
print("G3-C4 P2/G2 EOS first frame static PASS; predecessor bytes unchanged")
