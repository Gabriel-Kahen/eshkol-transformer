#!/usr/bin/env python3
"""Bounded source gate for the private Eshkol typed owner and carrier."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = "native/g3c4_p2g2_typed_result_extension.esk"
source = (root / path).read_text()
base = (root / "native/g3c4_p2g2_typed_result_source_closure.txt").read_text().splitlines()
added = [
    path,
    "scripts/check-g3c4-p2g2-typed-wrapper-source.py",
    "scripts/test-g3c4-p2g2-typed-wrapper-source.sh",
    "docs/g3/G3_C4_P2_G2_ESHKOL_TYPED_WRAPPER_LEAF.md",
    "tests/g3c4/p2_g2_typed_wrapper_rollback_test.scm",
    "scripts/test-g3c4-p2g2-typed-wrapper-host.sh",
    "native/g3c4_p2g2_typed_wrapper_source_closure.txt",
]
closure = (root / added[-1]).read_text().splitlines()
assert closure == base + added and len(closure) == len(set(closure))
assert all((root / name).is_file() for name in closure)

# Reject malformed source without requiring the pinned compiler in a host-only
# source gate. Strings and semicolon comments do not contribute delimiters.
depth = 0
quoted = escaped = commented = False
for char in source:
    if commented:
        if char == "\n":
            commented = False
        continue
    if quoted:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            quoted = False
        continue
    if char == ";":
        commented = True
    elif char == '"':
        quoted = True
    elif char == "(":
        depth += 1
    elif char == ")":
        depth -= 1
        assert depth >= 0
assert depth == 0 and not quoted

header = (root / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
native = (root / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
for symbol in (
    "et_g3c4_private_p2g2_typed_ids_create_v1",
    "et_g3c4_private_p2g2_typed_scalar_create_v1",
    "et_g3c4_private_p2g2_typed_length_v1",
    "et_g3c4_private_p2g2_typed_copy_v1",
    "et_g3c4_private_p2g2_typed_release_v1",
):
    assert symbol in source and symbol in header and symbol in native


def block(name, after=None):
    start = source.index(f"(define ({name}")
    end = source.find("\n(define (", start + 1)
    return source[start:end if end >= 0 else None]


create = block("g3c4-p2g2-typed-create-private")
copy = block("g3c4-p2g2-typed-copy-private")
release = block("g3c4-p2g2-typed-release-private!")
live = block("g3c4-p2g2-typed-entry-live")


def ordered(text, fragments):
    positions = [text.find(fragment) for fragment in fragments]
    return -1 not in positions and positions == sorted(positions)


def valid_create(text):
    cleanup = text.split("(guard (caught", 1)[1].split(
        "(vector-set! g3c4-registry 0 registry-node)", 1)[0]
    return (ordered(text, (
        "(source (g3c4-p2g2-terminal-ids-entry-live",
        "(source-bytes (vector-ref source 8))",
        "(bytes (make-bytevector payload 0))",
        "(old-registry (vector-ref g3c4-registry 0))",
        "(= (bytevector-length bytes) payload)",
        "(vector-set! g3c4-registry 0 registry-node)",
        "(set! native",
        "(anchor 'bind native)",
        "(g3c4-p2g2-native-typed-length native)",
        "(g3c4-p2g2-native-typed-copy native count bytes extent)",
        "(anchor 'match bytes)",
        "(anchor 'publish #f)",
        "(vector-set! entry 2 'live)",
    )) and ordered(cleanup, (
        "(g3c4-p2g2-native-typed-release",
        "(eq? (vector-ref g3c4-registry 0)",
        "(eq? (vector-ref g3c4-p2g2-typed-anchors 0)",
        "(g3c4-entry-dead! entry)",
        "(vector-set! g3c4-registry 0 old-registry)",
        "(vector-set! g3c4-p2g2-typed-anchors 0 old-anchors)",
        "(raise caught)",
    )) and "(g3c4-native-check\n            (g3c4-p2g2-native-typed-copy native count bytes extent)\n            operation)" in text and text.count("(= (bytevector-length bytes) payload)") == 3 and (
        text.count("(eq? source-bytes (vector-ref source 8))") == 2
    ) and all(item in text for item in (
        "(eq? source-native (vector-ref source 3))",
        "(eq? source-bytes (vector-ref source 8))",
        "(eq? bytes (vector-ref entry 8))",
        "(if (not (= (g3c4-p2g2-native-typed-release",
        "(g3c4-entry-dead! entry)",
        "(raise caught)",
    )))


def valid_release(text):
    return (ordered(text, (
        "(g3c4-p2g2-typed-entry-live shell operation)",
        "(barrier (vector #f))",
        "(anchor 'native-release live)",
        "(vector-set! barrier 0 #t)",
        "(g3c4-p2g2-zero-bytes! bytes (* count 8))",
        "(g3c4-entry-dead! live)",
        "(anchor 'retire live)",
    )) and "(if (vector-ref barrier 0) (exit 134))" in text)


def valid_copy(text):
    return ordered(text, (
        "(bytes (make-bytevector payload 0))",
        "(= (bytevector-length bytes) payload)",
        "(g3c4-p2g2-native-typed-copy native count bytes (+ 8 payload))",
        "(anchor 'match bytes)",
    )) and "(g3c4-native-check\n          (g3c4-p2g2-native-typed-copy native count bytes (+ 8 payload))\n          operation)" in text


assert valid_create(create) and valid_release(release)
assert "(anchor 'check entry)" in live
assert valid_copy(copy)
assert "(and retired? released? original-node" in create
assert "(g3c4-dead-entry? original-entry original-kind)" in create
for before, after in (
    ("(bytes (make-bytevector payload 0))", "(bytes #f)"),
    ("(= (bytevector-length bytes) payload)", "#t"),
    ("(eq? source-bytes (vector-ref source 8))", "#t"),
    ("(g3c4-p2g2-native-typed-copy native count bytes extent)", "0"),
    ("(anchor 'match bytes)", "#t"),
    ("(g3c4-entry-dead! entry)", "#t"),
    ("(vector-set! g3c4-registry 0 old-registry)", "#t"),
    ("(eq? (vector-ref g3c4-registry 0)", "(eq? old-registry"),
    ("(g3c4-native-check\n            (g3c4-p2g2-native-typed-copy native count bytes extent)\n            operation)",
     "(g3c4-p2g2-native-typed-copy native count bytes extent)"),
):
    assert valid_create(create.replace(before, after, 1)) is False
assert not valid_copy(copy.replace(
    "(g3c4-native-check\n          (g3c4-p2g2-native-typed-copy native count bytes (+ 8 payload))\n          operation)",
    "(g3c4-p2g2-native-typed-copy native count bytes (+ 8 payload))", 1))
for before, after in (
    ("(anchor 'native-release live)", "0"),
    ("(g3c4-p2g2-zero-bytes! bytes (* count 8))", "#t"),
    ("(g3c4-entry-dead! live)", "#t"),
    ("(anchor 'retire live)", "#t"),
):
    assert valid_release(release.replace(before, after, 1)) is False

assert "g3c4-p2g2-typed-result-extension" not in (
    root / "lib/transformer/generation.esk").read_text()
print("G3-C4 private Eshkol typed wrapper source PASS: mutations=14")
