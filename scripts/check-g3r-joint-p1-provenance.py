#!/usr/bin/env python3
"""Source-only mutation gate for the detached P1 preclaim capsule.

This does not prove that module-state-dict can clone under a live G3 pin.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
leaf = (root / "native/g3r_joint_p1_provenance_extension.esk").read_text()
binding = (root / "native/g3r_joint_original_owner_binding_extension.esk").read_text()
joint = (root / "native/g3r_joint_hold.c").read_text()
g3 = (root / "src/eshkol_transformer/g3t_transport.c").read_text()


def balanced(source: str) -> bool:
    depth = 0
    quoted = escaped = comment = False
    for c in source:
        if c == "\n":
            comment = False
        if comment:
            continue
        if quoted:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                quoted = False
        elif c == ";":
            comment = True
        elif c == '"':
            quoted = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            assert depth >= 0
    return depth == 0 and not quoted


def check(p: str, b: str, j: str, g: str) -> None:
    assert balanced(p) and balanced(b)
    topology = p[p.index("(define (g3r-joint-p1-topology-check!"):p.index(
        "(define (g3r-joint-p1-provenance!")]
    for required in ("state-dict-c2-parameter-topology-internal",
                     "(= index 14)", "m3t-paths index", "m3t-shapes index",
                     '((("head" "weight") ("token_embedding" "weight")))',
                     "(null? remaining-paths)", "(null? remaining-shapes)"):
        assert required in topology

    create = p[p.index("(define (g3r-joint-p1-provenance!"):p.index(
        "(define (g3r-joint-p1-provenance-check!")]
    assert create.count("(g3r-joint-original-binding-check!") == 2
    first = create.index("(g3r-joint-original-binding-check!")
    clone = create.index("(vector-set! ledger 2 (module-state-dict model))")
    second = create.index("(g3r-joint-original-binding-check!", first + 1)
    assert first < create.index("(guard (caught") < clone < second
    assert create.index("(g3r-joint-p1-topology-check!") < second
    assert "(state-dict-release! (vector-ref ledger 2))" in create
    assert "(vector-set! ledger 5 caught)" in create
    assert "(raise caught)" in create
    assert "(g3r-held-o2-fail-stop)" in create
    assert "g3r-joint-o2-end" not in p and "g3r-joint-o2-release-issued" not in p
    ownership = p[p.index("(define (g3r-joint-p1-provenance-record"):
                  p.index("(define (g3r-joint-p1-provenance-check!")]
    for required in ("g3r-joint-p1-provenance-registry",
                     "(eq? (vector-ref (car entries) 0) ledger)",
                     "(eq? record (vector-ref ledger 6))",
                     "(eq? (vector-ref record 1) (vector-ref ledger 1))",
                     "(eq? (vector-ref record 2) (vector-ref ledger 2))"):
        assert required in ownership
    assert "(vector-set! record 2 (vector-ref ledger 2))" in create
    assert "(set! g3r-joint-p1-provenance-registry enrollment)" in create
    release = p[p.index("(define (g3r-joint-p1-provenance-release!"):]
    assert "(g3r-joint-p1-provenance-owned-shell ledger operation)" in release
    assert "(if (not (vector-ref ledger 5))" in release
    assert "(state-dict-release! exact)" in release

    wrapper = b[b.index("(define (g3r-joint-c2-original-bound-claim!"):]
    for required in ("(g3r-joint-p1-provenance-check! p1-ledger operation)",
                     "(p1 (vector-ref provenance 2))",
                     "(g3r-joint-p1-provenance-release! p1-ledger)",
                     "(vector-ref bound 2)", "(vector-ref bound 4)",
                     "(vector-ref bound 7)", "(vector-ref bound 8)",
                     "(vector-ref bound 9)",
                     "(result (vector binding p1-ledger #f))"):
        assert required in wrapper
    assert wrapper.index("(g3r-joint-original-binding-check!") < wrapper.index(
        "(g3r-joint-c2-claim! o2-ledger p1 (vector-ref binding 9))")
    assert wrapper.index("(g3r-joint-c2-claim! o2-ledger p1 (vector-ref binding 9))") < wrapper.index(
        "(vector-set! p1-ledger 4 'c2-owned)")

    match = j[j.index("int64_t et_g3r_joint_hold_owner_match_v1("):j.index(
        "et_o2_optimizer_state *et_g3r_joint_hold_issue_v1(")]
    assert "joint_find(candidate)" in match
    assert "generator != hold->generator" in match
    assert "sealed_model != hold->model" in match
    assert "optimizer != hold->optimizer" in match
    assert "return et_g3r_joint_hold_check_v1(hold);" in match
    held = g[g.index("int64_t et_g3r_private_g3t_save_check_v1("):g.index(
        "int64_t et_g3r_private_g3t_save_copy_v1(")]
    assert "g3t_check_binding(c)" in held
    assert "memcmp(&now, &hold->value, sizeof(now))" in held


check(leaf, binding, joint, g3)
mutations = (
    (leaf.replace("(= index 14)", "(= index 13)"), binding, joint, g3),
    (leaf.replace('((("head" "weight") ("token_embedding" "weight")))', "'()"), binding, joint, g3),
    (leaf.replace("m3t-shapes index", "m3t-paths index"), binding, joint, g3),
    (leaf.replace("(module-state-dict model)", "(module-state-dict other)"), binding, joint, g3),
    (leaf.replace("(state-dict-release! (vector-ref ledger 2))", "#t"), binding, joint, g3),
    (leaf.replace("(raise caught)", "#t"), binding, joint, g3),
    (leaf.replace("(g3r-joint-original-binding-check!", "(always-pass", 1), binding, joint, g3),
    (leaf.replace("(g3r-joint-original-binding-check!", "(always-pass"), binding, joint, g3),
    (leaf.replace("(eq? (vector-ref record 2) (vector-ref ledger 2))", "#t"), binding, joint, g3),
    (leaf.replace("(set! g3r-joint-p1-provenance-registry enrollment)", "#t"), binding, joint, g3),
    (leaf.replace("(if (not (vector-ref ledger 5))", "(if #t"), binding, joint, g3),
    (leaf, binding.replace("(g3r-joint-p1-provenance-release! p1-ledger)", "#t"), joint, g3),
    (leaf, binding.replace("(vector-ref bound 8)", "(vector-ref binding 8)"), joint, g3),
    (leaf, binding.replace("(g3r-joint-c2-claim! o2-ledger p1 (vector-ref binding 9))", "#f"), joint, g3),
    (leaf, binding, joint.replace("sealed_model != hold->model", "sealed_model == hold->model"), g3),
    (leaf, binding, joint, g3.replace("memcmp(&now, &hold->value, sizeof(now))", "0")),
)
for changed in mutations:
    try:
        check(*changed)
    except (AssertionError, ValueError):
        continue
    raise AssertionError("P1 provenance mutation escaped source gate")

assert "g3r_joint_p1_provenance_extension.esk" not in (
    (root / "Makefile").read_text()
)
print("joint P1 provenance source gate: pass (16 mutation controls; unlinked)")
