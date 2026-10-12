#!/usr/bin/env python3
"""Check the private compiled aggregate load and real-route test contract."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
subprocess.run(
    ["python3", str(root / "scripts/check-g3r-committed-save-snapshot.py")],
    check=True,
)
fixture = (root / "tests/g3r/committed_save_aggregate_test.esk").read_text()
runner = (root / "scripts/test-g3r-committed-save-aggregate.sh").read_text()
assert fixture.count('(load "g3r_committed_save_snapshot_extension.esk")') == 1
assert fixture.index('(load "g3m_seeded_p1_g1_extension.esk")') < fixture.index(
    '(load "g3r_committed_save_snapshot_extension.esk")'
)
for required in (
    "(g3m-generate-p1-g1! generator input)",
    "(g3r-committed-save-snapshot! generator bytes)",
    "(g3r-test-cache-length native)",
    "(g3r-test-rng-word native index)",
    "(g3t-output-release! output)",
    "(g3r-test-cache-borrow-begin native)",
    "(g3r-native-call-acquire native 0)",
    "(g3r-test-dispatch-fail-after 21)",
    "(all-sentinel? bytes)",
):
    assert required in fixture, required
for required in (
    "ET_G3R_COMMITTED_HISTORY_PRIVATE",
    "ET_G3R_COMMITTED_SAVE_SNAPSHOT_PRIVATE",
    "ET_F32_TENSOR_STORAGE_QUERY_PRIVATE",
    "ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE",
    "committed_save_aggregate_test.esk",
    "build_mode normal", "build_mode sanitize", "repeat.stdout",
    "feature-off.o", "nm -g --defined-only",
):
    assert required in runner, required
for public in (
    "native/g3g_package_root.esk", "native/g3g_manual_package_root.esk",
    "native/cli3_generate_root.esk",
):
    assert "g3r_committed_save_snapshot_extension.esk" not in (root / public).read_text()
print("G3-R committed save aggregate source contract PASS")
