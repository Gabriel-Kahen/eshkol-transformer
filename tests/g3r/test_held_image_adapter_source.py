"""Bounded source closure and mutation gate for the noninstalled adapter.

The Eshkol aggregate itself still needs the pinned compiled runtime gate.
"""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
source = (root / "native/g3r_held_image_adapter_extension.esk").read_text()
pure = (root / "native/g3r_committed_save_snapshot_extension.esk").read_text()
bridge = (root / "native/g3r_held_image_bridge.c").read_text()
root_source = (root / "native/g3r_held_image_adapter_root.esk").read_text()
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
native_test = (root / "tests/g3r/current_history_held_owner_test.c").read_text()
fixture = (root / "tests/g3r/held_image_adapter_runtime.esk").read_text()


def balanced(text):
    depth = 0
    quote = escape = comment = False
    for c in text:
        if comment:
            if c == "\n":
                comment = False
        elif quote:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                quote = False
        elif c == ";":
            comment = True
        elif c == '"':
            quote = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            assert depth >= 0
    assert depth == 0 and not quote


balanced(source)
balanced(fixture)
assert root_source.count('(load "g3r_original_owner_c2_claim_root.esk")') == 1
assert root_source.count('(load "g3r_committed_save_snapshot_extension.esk")') == 1
assert root_source.count('(load "g3r_held_image_adapter_extension.esk")') == 1
assert source.count("(define (g3r-held-image-project!") == 1
assert "(define g3r-held-image-copy-internal g3r-held-image-native-copy)" in source
assert fixture.count('(load "g3r_held_image_adapter_root.esk")') == 1
assert fixture.count("(load ") == 1
for needle in (
    "(g3r-original-owner-begin!", "(g3r-held-o2-adopt!",
    "(c2-training-state-compose-internal request)",
    "(g3r-held-o2-adopter-c2-claim!", "(define (cut-bridge",
    "(define (cut-recheck", "(define (cut-image",
    "(define (cut-c2-control", "(raise #f)",
    "(c2-training-state-borrow-begin-internal",
    "(c2-training-state-borrow-end-internal",
    "(g3r-original-owner-end!", "(c2-training-state-release-internal!",
    '"uncertain-release-raise"', '"uncertain-release-return"',
    "(current-error-port)",
    "(check \"failure publishes no seven fields\"",
    "(retry-project transfer shell)",
):
    assert needle in fixture, needle

owner = source.split("(define (g3r-held-image-owner-recheck!", 1)[1].split(
    "(define (g3r-held-image-controls-check!", 1)[0]
controls = source.split("(define (g3r-held-image-controls-check!", 1)[1].split(
    "(define (g3r-held-image-project!", 1)[0]
project = source.split("(define (g3r-held-image-project!", 1)[1]
native_copy = native.split("int64_t et_g3r_private_g3t_save_copy_v1(", 1)[1].split(
    "int64_t et_g3r_private_g3t_save_end_v1", 1)[0]


def contract(owner_text, controls_text, project_text, pure_text, bridge_text):
    for needle in (
        "g3r-original-owner-tag", "'held", "g3r-trainer-g3-save-tag",
        "tr3-lease-record-for-shell-internal",
        "tr3-lease-snapshot-recheck-held-o2-internal",
        "g3r-original-owner-g3-recheck!", "g3r-trainer-g3-native-check",
    ):
        assert needle in owner_text, needle
    assert owner_text.count("tr3-lease-snapshot-recheck-held-o2-internal") == 2
    for needle in (
        "g3r-save-tokenizer?", "g3r-save-tokenizer-fingerprint",
        "(vector fingerprint 256)", "(vector-ref record 18)",
        "(vector-ref record 17)", "(vector-ref record 19)",
        "(vector-ref record 20)", "(vector-ref record 21)",
        "x1-private-config-canonical", "x1-private-config-fingerprint",
        "x1-canonical-inspect-internal",
    ):
        assert needle in controls_text, needle
    for needle in (
        "g3r-held-o2-adopter-tag", "'claimed", "c2-training-state-find",
        "c2-training-state-borrow-begin-internal",
        "c2-training-state-borrow-projection-internal",
        "g3r-held-image-copy-internal hold staging",
        "g3r-save-data staging entry",
        "c2-training-state-borrow-end-internal",
        "tr3-lease-fail-stop 134", "(raise caught)",
        "(eq? #t",
    ):
        assert needle in project_text, needle
    assert project_text.count("g3r-held-image-owner-recheck! original operation") == 4
    assert project_text.count("g3r-held-image-controls-check! controls record entry operation") == 2
    assert project_text.index("c2-training-state-borrow-begin-internal") < project_text.index(
        "g3r-held-image-copy-internal hold staging") < project_text.index(
        "g3r-save-data staging entry") < project_text.rindex(
        "c2-training-state-borrow-end-internal") < project_text.rindex(
        "g3r-held-image-owner-recheck! original operation")
    assert "(make-bytevector g3r-save-size 0)" in project_text
    assert "(vector-ref result-cell 0)" in project_text
    for needle in (
        "(g3r-save-words bytes 0 6)", "(g3r-save-words bytes 6 4)",
        "(g3r-save-word bytes 12)", "(g3r-save-word bytes 16)",
        "(vector policy rng history prompt generated manual terminal)",
        "(= terminal", "(if (and (= generated 1)",
    ):
        assert needle in pure_text, needle
    assert "length != (int64_t)sizeof(et_g3r_g3t_held_words_v1)" in bridge_text
    assert "et_g3r_private_g3t_save_copy_v1(hold, image, length)" in bridge_text


contract(owner, controls, project, pure, bridge)
assert "et_g3r_private_g3t_save_check_v1(hold)" in native_copy
assert "memcpy(destination, &hold->value, sizeof(*destination))" in native_copy
for needle in ("history_pending_g1 = 1", "history_ids[0] ^= 1",
               "model_bits[0] = 1", "c.rng[2] ^= 1", "c.policy[2] = 0",
               "c.history_generated_eos_terminal = 1",
               "c.policy[0] ^= 1", "held_words_before"):
    assert needle in native_test

# Each cut removes one causal authority, ordering, provenance, or cleanup
# edge. The source gate must fail on every corresponding mutant.
cuts = [
    ("owner", "g3r-original-owner-g3-recheck!"),
    ("owner", "g3r-trainer-g3-native-check"),
    ("controls", "g3r-save-tokenizer?"),
    ("controls", "(vector fingerprint 256)"),
    ("controls", "(vector-ref record 18)"),
    ("controls", "(vector-ref record 17)"),
    ("controls", "(vector-ref record 19)"),
    ("project", "g3r-held-image-copy-internal hold staging"),
    ("project", "g3r-save-data staging entry"),
    ("project", "tr3-lease-fail-stop 134"),
    ("pure", "(g3r-save-word bytes 16)"),
    ("bridge", "et_g3r_private_g3t_save_copy_v1(hold, image, length)"),
]
parts = dict(owner=owner, controls=controls, project=project,
             pure=pure, bridge=bridge)
for section, needle in cuts:
    assert needle in parts[section]
    changed = dict(parts)
    changed[section] = changed[section].replace(needle, "CUT")
    try:
        contract(changed["owner"], changed["controls"], changed["project"],
                 changed["pure"], changed["bridge"])
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError(f"source mutant escaped: {section}: {needle}")

print(f"G3R HELD IMAGE SOURCE PASS: {len(cuts)} causal cuts")
