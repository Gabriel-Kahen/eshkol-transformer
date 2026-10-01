#!/usr/bin/env python3
"""Static boundary for the private P2/G2 Eshkol carrier bridge."""

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
subprocess.run(
    [sys.executable, str(root / "scripts/check-g3c4-p2g2-prefix-commit.py")],
    cwd=root, check=True, stdout=subprocess.DEVNULL,
)
owner = (root / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
header = (root / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
test = (root / "tests/g3c4/test_p2_g2_carrier_bridge.c").read_text()
runner = (root / "scripts/test-g3c4-p2g2-carrier-bridge.sh").read_text()
contract = (root / "docs/g3/G3_C4_P2_G2_CARRIER_BRIDGE_LEAF.md").read_text()
for required in (
    "C4 P2/G2 carrier bridge requires prefix commit and private f32 storage inspection",
    "et_g3c4_private_p2g2_first_frame_carrier_v1(",
    "et_g3c4_p2g2_carrier_owned_alias(",
    "et_f32_tensor_private_storage_overlap_v1(",
    "carrier_bytes != (int64_t)(sizeof(int64_t) + sizeof(encoded))",
    "declared_bytes != (int64_t)sizeof(encoded)",
    "et_g3c4_private_prompt_prefill_v1(",
    "et_g3c4_private_token_frame_begin_last_v1(",
    "et_g3c4_private_token_forward_v1(",
    "(unsigned char *)staging_header + sizeof(declared_bytes)",
):
    if required not in owner:
        raise SystemExit(f"P2/G2 bridge source contract missing: {required}")
if "ET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE" not in header:
    raise SystemExit("P2/G2 bridge private declaration missing")
for required in (
    "parity_case(owner, (const int64_t[2]){0, 255}, 0)",
    "parity_case(owner, (const int64_t[2]){7, 11}, 1)",
    "fail_prefill2_at = cut",
    "forged_token = mode == 1 ? -1 : 256",
    "fail_bridge_forward = 1",
    "check_carrier_untouched(&carrier)",
    "check_cache_snapshot_equal(&actual_cache, &reference_cache)",
    "context, dead_input, &carrier, 16",
    "context, input, (void *)keys->data, 16",
    "context, input, (void *)f32_data, 16",
    "context->generator_policy[5] = 56",
    "context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED",
    "carrier.canary[i] == 0x5au",
):
    if required not in test:
        raise SystemExit(f"P2/G2 bridge test contract missing: {required}")
for required in (
    'comm -3 "$tmp/off.symbols" "$tmp/on.symbols"',
    "et_g3c4_private_p2g2_first_frame_carrier_v1",
    "isolated P2/G2 bridge macro compiled",
    "-DET_F32_TENSOR_STORAGE_QUERY_PRIVATE",
    "check_p2_g1_oracle.py",
):
    if required not in runner:
        raise SystemExit(f"P2/G2 bridge runner contract missing: {required}")
for required in (
    "bytevector-length==8", "backing extent", "eight little-endian bytes",
    "not that coordinator", "source-private",
):
    if required not in contract:
        raise SystemExit(f"P2/G2 bridge contract missing: {required}")
print("G3-C4 P2/G2 carrier bridge static PASS")
