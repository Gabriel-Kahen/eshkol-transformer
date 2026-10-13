#!/usr/bin/env python3
"""Verify the adapter remains source-private and composed from admitted entries."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
source = (root / "native/g3r_snapshot_codec_adapter_extension.esk").read_text()
fixture = (root / "tests/g3r/snapshot_codec_adapter_aggregate_test.esk").read_text()
runner = (root / "scripts/test-g3r-snapshot-codec-adapter-aggregate.sh").read_text()
for required in ("g3r-committed-save-snapshot!", "g3r-record-encode!",
                 "g3r-snapshot-record-fields?", "bytevector-copy! destination",
                 "(eq? destination record-scratch)"):
    assert required in source, required
for name in ("g3r_committed_save_snapshot_extension.esk", "c1_sha256.esk",
             "g3r_record_codec_extension.esk", "g3r_snapshot_codec_adapter_extension.esk"):
    assert fixture.count(f'(load "{name}")') == 1, name
assert fixture.index('g3r_committed_save_snapshot_extension.esk') < fixture.index('g3r_snapshot_codec_adapter_extension.esk')
for required in ("build_mode normal", "build_mode sanitize", "repeat.stdout",
                 "check_snapshot_codec_adapter_oracle.py", "feature-off.o"):
    assert required in runner, required
for public in ("native/g3g_package_root.esk", "native/g3g_manual_package_root.esk",
               "native/cli3_generate_root.esk"):
    assert "g3r_snapshot_codec_adapter_extension.esk" not in (root / public).read_text()
print("G3-R snapshot codec adapter source contract PASS")
