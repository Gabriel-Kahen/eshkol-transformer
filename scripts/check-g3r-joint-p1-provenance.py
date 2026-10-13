#!/usr/bin/env python3
"""Source-only mutation gate for the detached P1 preclaim capsule.

This does not prove that module-state-dict can clone under a live G3 pin.
"""
from pathlib import Path
from copy import deepcopy

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
    witness = p[p.index("(define (g3r-joint-p1-owner-witness-copy"):
                p.index("(define (g3r-joint-p1-topology-check!")]
    assert "(c2-training-state-controls-copy operation (vector-ref binding 9))" in witness
    assert "(eq? (vector-ref witness 0) g3r-joint-p1-owner-witness-tag)" in witness
    assert "(eq? (vector-ref binding 0) g3r-joint-original-binding-tag)" in witness
    assert "(eq? (vector-ref witness index)" in witness
    assert "(vector-ref binding index)" in witness
    assert "(equal? (vector-ref witness 9) (vector-ref binding 9))" in witness
    topology = p[p.index("(define (g3r-joint-p1-topology-check!"):p.index(
        "(define (g3r-joint-p1-provenance!")]
    for required in ("state-dict-c2-parameter-topology-internal",
                     "(= index 14)", "m3t-paths index", "m3t-shapes index",
                     '((("head" "weight") ("token_embedding" "weight")))',
                     "(null? remaining-paths)", "(null? remaining-shapes)"):
        assert required in topology

    create = p[p.index("(define (g3r-joint-p1-provenance!"):p.index(
        "(define (g3r-joint-p1-provenance-record")]
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
    assert "(ledger (vector g3r-joint-p1-provenance-tag binding #f #f\n                         'prepared #f))" in create
    assert "(witness (g3r-joint-p1-owner-witness-copy operation binding))" in create
    assert "(record (vector ledger witness #f 'prepared))" in create
    assert "(g3r-joint-p1-owner-witness-match? witness rechecked)" in create
    assert "(vector-set! ledger 6 record)" not in p
    ownership = p[p.index("(define (g3r-joint-p1-provenance-record"):
                  p.index("(define (g3r-joint-p1-provenance-check!")]
    for required in ("g3r-joint-p1-provenance-registry",
                     "(eq? (vector-ref (car entries) 0) ledger)",
                     "(eq? (vector-ref record 3) 'preclaim)",
                     "(g3r-joint-p1-owner-witness-match?",
                     "(vector-ref record 1) (vector-ref ledger 1)",
                     "(eq? (vector-ref record 2) (vector-ref ledger 2))",
                     "(vector-ref record 2)))"):
        assert required in ownership
    assert "(vector-set! record 2 (vector-ref ledger 2))" in create
    assert "(vector-set! record 3 'preclaim)" in create
    assert "(set! g3r-joint-p1-provenance-registry enrollment)" in create
    release = p[p.index("(define (g3r-joint-p1-provenance-release!"):]
    assert "(record (g3r-joint-p1-provenance-live-record ledger))" in release
    assert "(if (not (vector-ref ledger 5))" in release
    assert "(state-dict-release! (vector-ref record 2))" in release
    assert "(vector-set! record 3 'released)" in release

    wrapper = b[b.index("(define (g3r-joint-c2-original-bound-claim!"):]
    for required in ("(g3r-joint-p1-provenance-check! p1-ledger operation)",
                     "(record (g3r-joint-p1-provenance-live-record p1-ledger))",
                     "(p1 (vector-ref record 2))",
                     "(bound (vector-ref record 1))",
                     "(g3r-joint-p1-provenance-release! p1-ledger)",
                     "(g3r-joint-p1-owner-witness-match? bound binding)",
                     "(result (vector binding p1-ledger #f))"):
        assert required in wrapper
    assert wrapper.index("(record (g3r-joint-p1-provenance-live-record p1-ledger))") < wrapper.index(
        "(guard (caught") < wrapper.index("(g3r-joint-p1-provenance-check! p1-ledger operation)")
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

# A bounded data-level negative for the rereviewed alias attack. The source
# assertions above bind this model to the actual private copy/match operations;
# this is still not an Eshkol runtime test.
owner_a = [object() for _ in range(8)]
owner_b = [object() for _ in range(8)]
binding_a = ["binding", *owner_a, [["A"]]]
private_witness = ["witness", *binding_a[1:9], deepcopy(binding_a[9])]
shared_old_witness = binding_a
binding_a[1:9] = owner_b
binding_a[9][0][0] = "B"
assert all(shared_old_witness[i] is binding_a[i] for i in range(1, 9))
assert all(private_witness[i] is not binding_a[i] for i in range(1, 9))
assert private_witness[9] != binding_a[9]

mutations = (
    (leaf.replace("(= index 14)", "(= index 13)"), binding, joint, g3),
    (leaf.replace('((("head" "weight") ("token_embedding" "weight")))', "'()"), binding, joint, g3),
    (leaf.replace("m3t-shapes index", "m3t-paths index"), binding, joint, g3),
    (leaf.replace("(module-state-dict model)", "(module-state-dict other)"), binding, joint, g3),
    (leaf.replace("(state-dict-release! (vector-ref ledger 2))", "#t"), binding, joint, g3),
    (leaf.replace("(raise caught)", "#t"), binding, joint, g3),
    (leaf.replace("(g3r-joint-original-binding-check!", "(always-pass", 1), binding, joint, g3),
    (leaf.replace("(g3r-joint-original-binding-check!", "(always-pass"), binding, joint, g3),
    # Reintroduce the rereviewed cross-owner path: shared mutable binding
    # or shared nested controls lets caller edits retarget the P1 witness.
    (leaf.replace("(witness (g3r-joint-p1-owner-witness-copy operation binding))",
                  "(witness binding)"), binding, joint, g3),
    (leaf.replace("(record (vector ledger witness #f 'prepared))",
                  "(record (vector ledger binding #f 'prepared))"), binding, joint, g3),
    (leaf.replace("(c2-training-state-controls-copy operation (vector-ref binding 9))",
                  "(vector-ref binding 9)"), binding, joint, g3),
    (leaf.replace("(eq? (vector-ref witness index)", "(eq? (vector-ref binding index)"),
     binding, joint, g3),
    (leaf.replace("(eq? (vector-ref record 2) (vector-ref ledger 2))", "#t"), binding, joint, g3),
    (leaf.replace("'prepared #f))\n         (record", "'prepared #f #f))\n         (record").replace(
        "(guard (caught", "(vector-set! ledger 6 record)\n    (guard (caught", 1), binding, joint, g3),
    (leaf.replace("(set! g3r-joint-p1-provenance-registry enrollment)", "#t"), binding, joint, g3),
    (leaf.replace("(if (not (vector-ref ledger 5))", "(if #t"), binding, joint, g3),
    (leaf, binding.replace("(g3r-joint-p1-provenance-release! p1-ledger)", "#t"), joint, g3),
    (leaf, binding.replace("(p1 (vector-ref record 2))", "(p1 (vector-ref p1-ledger 2))"), joint, g3),
    (leaf, binding.replace("(g3r-joint-p1-provenance-check! p1-ledger operation)", "#t"), joint, g3),
    (leaf, binding.replace("(g3r-joint-p1-owner-witness-match? bound binding)", "#t"), joint, g3),
    (leaf, binding.replace("(guard (caught", "(g3r-joint-p1-provenance-check! p1-ledger operation)\n    (guard (caught", 1).replace(
        "      (g3r-joint-p1-provenance-check! p1-ledger operation)\n      (let*", "      (let*"), joint, g3),
    (leaf, binding.replace("(bound (vector-ref record 1))",
                           "(bound (vector-ref p1-ledger 1))"), joint, g3),
    (leaf, binding.replace("(g3r-joint-c2-claim! o2-ledger p1 (vector-ref binding 9))", "#f"), joint, g3),
    (leaf, binding, joint.replace("sealed_model != hold->model", "sealed_model == hold->model"), g3),
    (leaf, binding, joint, g3.replace("memcmp(&now, &hold->value, sizeof(now))", "0")),
)
for index, changed in enumerate(mutations):
    try:
        check(*changed)
    except (AssertionError, ValueError):
        continue
    raise AssertionError(f"P1 provenance mutation {index} escaped source gate")

assert "g3r_joint_p1_provenance_extension.esk" not in (
    (root / "Makefile").read_text()
)
print("joint P1 provenance source gate: pass (25 mutation controls; unlinked)")
