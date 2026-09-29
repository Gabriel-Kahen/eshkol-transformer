#!/usr/bin/env python3
"""Bounded source-private P2/G1 shell publication closure check."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
shell = (root / "native/g3c4_p2g1_shell_publication_extension.esk").read_text()
fixture = "\n".join(
    (root / f"tests/g3c4/p2_g1_shell_{case}_test.esk").read_text()
    for case in ("success", "mutation", "lease", "owner_swap")
)
common = (root / "tests/g3c4/p2_g1_shell_test_common.esk").read_text()
edge = (root / "tests/g3c4/p2_g1_shell_edge_escape_test.esk").read_text()
runner = (root / "scripts/test-g3c4-p2g1-shell.sh").read_text()
contract = (root / "docs/g3/G3_C4_P2_G1_SHELL_PUBLICATION_PROPOSAL.md").read_text()
public = (root / "lib/transformer/generation.esk").read_text()

for phrase in (
    "g3c4-with-p2g1-output-call-internal",
    "g3c4-p2g1-output-publish!",
    "g3c4-p2g1-output-entry-live",
    "g3c4-p2g1-output-release-internal!",
    "g3c4-native-generation-frame-prepare",
    "g3c4-native-generation-frame-commit",
    "g3c4-native-call-prepare-end",
    "g3c4-native-call-finish",
    "g3c4-p2g1-zero-bytes!",
    "accepted-byte",
):
    assert phrase in shell, phrase
assert shell.index("g3t-native-output-accept-text") < shell.index(
    "g3c4-native-generation-frame-prepare native output-native"
) < shell.index("g3c4-native-call-prepare-end native") < shell.index(
    "g3c4-native-generation-frame-commit native output-native"
) < shell.index("g3c4-native-call-finish native")
assert "(vector-set! ledger 3 'committed)" in shell
assert "g3c4-with-call-internal" not in shell
assert "g3c4-p2g1-output-publish!" not in public
for phrase in (
    "run-success 'greedy",
    "run-success 'categorical",
    "(run-mutation 3)",
    "fresh retry after abort publishes",
    "held lease rejects publication in active region",
    "altered child ledger rejects before rollback",
    "borrowed output release preserves live shell",
    "swapped native owner cannot release another output",
):
    assert phrase in fixture, phrase
assert "--wrap=et_g3c4_private_output_accept_text_v1" in runner
assert "g3c4-p2g1-output-reserve" in common
assert "p2_g1_shell_${test}_test.esk" in runner
assert "p2_g1_shell_edge_escape_test.esk" in runner
assert "edge_status" in runner and "-eq 134" in runner
assert "(vector-set! (vector-ref call 10) 4 #f)" in edge
assert "normal" in runner and "sanitize" in runner
assert "no raw alias" in contract
print("G3-C4 P2/G1 shell static PASS: private coordinator and cuts")
