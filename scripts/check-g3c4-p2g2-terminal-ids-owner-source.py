#!/usr/bin/env python3
"""Source-only closure and failure-boundary checks for private logical I1 clones."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "native/g3c4_p2g2_terminal_ids_owner_extension.esk").read_text()
base = (root / "native/g3c4_p2g2_terminal_release_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_terminal_ids_owner_extension.esk",
    "scripts/check-g3c4-p2g2-terminal-ids-owner-source.py",
    "scripts/test-g3c4-p2g2-terminal-ids-owner-source.sh",
    "docs/g3/G3_C4_P2_G2_LOGICAL_I1_CLONE_CONTRACT.md",
    "native/g3c4_p2g2_terminal_ids_owner_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions and len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure)
header = (root / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
native = (root / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
for symbol in (
    "et_g3c4_private_p2g2_terminal_ids_clone_v1",
    "et_g3c4_private_p2g2_terminal_ids_clone_copy_v1",
    "et_g3c4_private_p2g2_terminal_ids_clone_release_v1",
):
    assert symbol in header and symbol in native
assert "#ifdef ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE" in header


def valid(text):
    code = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    depth = 0
    for char in code:
        depth += (char == "(") - (char == ")")
        if depth < 0:
            return False
    if depth != 0:
        return False
    required = (
        ":real et_g3c4_private_p2g2_terminal_ids_clone_v1",
        ":real et_g3c4_private_p2g2_terminal_ids_clone_copy_v1",
        ":real et_g3c4_private_p2g2_terminal_ids_clone_release_v1",
        "(g3c4-p2g2-terminal-snapshot-private shell)",
        "(g3c4-p2g2-terminal-output-entry-live",
        "(g3c4-p2g2-terminal-snapshot-live-for",
        "(bytevector-u8-set! bytes 0 payload)",
        "(vector-set! entry 10 anchor)",
        "(vector-set! g3c4-registry 0 registry-node)",
        "(vector-set! g3c4-p2g2-terminal-ids-anchors 0 anchor-node)",
        "(g3c4-p2g2-native-terminal-ids-clone source-native)",
        "(anchor 'bind created)",
        "(g3c4-p2g2-native-terminal-ids-clone-copy",
        "created count bytes extent",
        "(anchor 'bytes-match bytes)",
        "(anchor 'bound created)",
        "(eq? (vector-ref entry 8) bytes)",
        "(= (bytevector-length bytes) payload)",
        "(= (bytevector-u8-ref bytes 0) payload)",
        "(g3c4-p2g2-terminal-snapshot-small-word",
        "(anchor 'publish #f)",
        "(vector-set! entry 2 'live)",
        "(g3c4-p2g2-terminal-ids-entry-live shell operation)",
        "(g3c4-p2g2-native-terminal-ids-clone-release\n                                original-native)",
        "(g3c4-dead-entry? entry 'terminal-ids)",
        "(begin\n              (g3c4-p2g2-terminal-ids-entry-dead shell entry operation)\n              #t)",
        "(g3c4-p2g2-terminal-ids-entry-dead shell entry operation)",
        "(let ((anchor (g3c4-p2g2-terminal-ids-anchor-for shell operation)))",
        "(anchor 'dead entry)",
        "(anchor 'native-release release-request)",
        "(release-request (cons live barrier))",
        "(if (= status 0)\n                            (begin\n                              (vector-set! (cdr candidate) 0 #t)\n                              (set! native-released? #t)))",
        "(and retired? native-released? node-bound?",
        "(anchor 'retire live)",
        "(anchor 'node-bind anchor-node)",
        "(anchor 'node nodes)",
        "(g3c4-p2g2-zero-bytes! bytes (* count 8))",
        "(g3c4-entry-dead! live)",
    )
    if any(fragment not in code for fragment in required):
        return False
    if any(fragment in code for fragment in (
        "g3c4-public-generation",
        "g3c4-native-result-tensor-create",
        "g3c4-p2g1-output-release-internal!",
    )):
        return False
    for repeated in (
        "(anchor 'bytes-match bytes)",
        "(g3c4-p2g2-terminal-ids-entry-live shell operation)",
        "(eq? (vector-ref entry 8) bytes)",
        "(= (bytevector-length bytes) payload)",
    ):
        if code.count(repeated) != 2:
            return False
    clone = code.index("(g3c4-p2g2-native-terminal-ids-clone source-native)")
    copy = code.index("created count bytes extent")
    match = code.rindex("(anchor 'bytes-match bytes)")
    publish = code.index("(anchor 'publish #f)")
    live = code.index("(vector-set! entry 2 'live)")
    release = code.index("(define (g3c4-p2g2-terminal-ids-release-private! shell)")
    dead = code.index("(g3c4-p2g2-terminal-ids-entry-dead shell entry operation)", release)
    native_release = code.index("(anchor 'native-release release-request)", release)
    anchor_release = code.index("((eq? command 'native-release)")
    native_status = code.index("(g3c4-p2g2-native-terminal-ids-clone-release", anchor_release)
    barrier = code.index("(vector-set! (cdr candidate) 0 #t)", anchor_release)
    recorded = code.index("(set! native-released? #t)", anchor_release)
    scrub = code.index("(g3c4-entry-dead! live)", release)
    retire = code.index("(anchor 'retire live)", release)
    return (clone < copy < match < publish < live < release
            and anchor_release < native_status < barrier < recorded < release
            and release < dead < native_release < scrub < retire)


assert valid(source)
mutations = (
    ("et_g3c4_private_p2g2_terminal_ids_clone_v1",
     "et_g3c4_private_p2g2_terminal_ids_clone_wrong"),
    ("et_g3c4_private_p2g2_terminal_ids_clone_copy_v1",
     "et_g3c4_private_p2g2_terminal_ids_clone_copy_wrong"),
    ("et_g3c4_private_p2g2_terminal_ids_clone_release_v1",
     "et_g3c4_private_p2g2_terminal_ids_clone_release_wrong"),
    ("(g3c4-p2g2-terminal-snapshot-private shell)", "#f"),
    ("(g3c4-p2g2-native-terminal-ids-clone source-native)", "'()"),
    ("created count bytes extent", "created count bytes payload"),
    ("(anchor 'bind created)", "#t"),
    ("(anchor 'bound created)", "#t"),
    ("(anchor 'bytes-match bytes)", "#t"),
    ("(eq? (vector-ref entry 8) bytes)", "#t"),
    ("(= (bytevector-length bytes) payload)", "#t"),
    ("(= (bytevector-u8-ref bytes 0) payload)", "#t"),
    ("(g3c4-p2g2-terminal-ids-entry-live shell operation)",
     "(g3c4-registry-find shell operation)"),
    ("(begin\n              (g3c4-p2g2-terminal-ids-entry-dead shell entry operation)\n              #t)", "#t"),
    ("(anchor 'node-bind anchor-node)", "#t"),
    ("(g3c4-p2g2-zero-bytes! bytes (* count 8))", "#t"),
    ("(anchor 'retire live)", "#t"),
    ("(vector-set! (cdr candidate) 0 #t)", "#t"),
    ("(g3c4-entry-dead! live)", "#t"),
    # A live owner whose mutable slot 2 was forged to 'dead must fail the
    # original-anchor proof rather than skip native release and cleanup.
    ("(anchor 'dead entry)", "#t"),
    ("(and retired? native-released? node-bound?",
     "(and retired? node-bound?"),
    ("(if (= status 0)\n                            (begin\n                              (vector-set! (cdr candidate) 0 #t)\n                              (set! native-released? #t)))",
     "(begin\n                              (vector-set! (cdr candidate) 0 #t)\n                              (set! native-released? #t))"),
    ("(anchor 'native-release release-request)",
     "(g3c4-p2g2-native-terminal-ids-clone-release native)"),
    # Re-labeling another kind's mutable fields cannot create this anchor.
    ("(let ((anchor (g3c4-p2g2-terminal-ids-anchor-for shell operation)))",
     "(let ((anchor #t))"),
)
for before, after in mutations:
    assert before in source and not valid(source.replace(before, after, 1)), before
print(f"G3-C4 terminal private I1 owner source PASS: {len(mutations)} negative mutations")
