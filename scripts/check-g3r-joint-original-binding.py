#!/usr/bin/env python3
"""Causal source gate for exact G3/T1/O2 original-owner control binding."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
native = (root / "native/g3r_joint_hold.c").read_text()
scheme = (root / "native/g3r_joint_original_owner_binding_extension.esk").read_text()
g3 = (root / "native/g3t_model_admission_extension.esk").read_text()
p1 = (root / "internal/p1/lib/transformer/module.esk").read_text()


def balanced(source: str) -> bool:
    depth = 0
    quote = escape = comment = False
    for char in source:
        if char == "\n":
            comment = False
        if comment:
            continue
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                quote = False
        elif char == ";":
            comment = True
        elif char == '"':
            quote = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            assert depth >= 0
    return depth == 0 and not quote


def check(n: str, s: str, g: str, p: str) -> None:
    assert balanced(s)
    finder = n[n.index("static et_g3r_joint_hold *joint_find("):
               n.index("et_g3r_joint_hold *et_g3r_joint_hold_begin_v1(")]
    assert "hold == candidate && hold->live == 1u" in finder
    match = n[n.index("int64_t et_g3r_joint_hold_owner_match_v1("):
              n.index("et_o2_optimizer_state *et_g3r_joint_hold_issue_v1(")]
    assert "joint_find(candidate)" in match
    for field in ("generator != hold->generator", "sealed_model != hold->model",
                  "optimizer != hold->optimizer"):
        assert field in match
    assert match.index("generator != hold->generator") < match.index(
        "et_g3r_joint_hold_check_v1(hold)")
    assert ":real et_g3r_joint_hold_owner_match_v1" in s
    binding = s[s.index("(define (g3r-joint-original-binding-check!"):
                s.index("(define (g3r-joint-c2-original-bound-claim!")]
    for field in ("(vector-ref generator-entry 5) model-entry",
                  "(vector-ref generator-entry 6) tokenizer",
                  "(vector-ref o2-ledger 2) optimizer-value",
                  "(vector-ref o2-ledger 3)",
                  "(vector-ref o2-ledger 8) 'enrolled",
                  "(g3r-save-tokenizer? tokenizer)",
                  "g3r-save-tokenizer-fingerprint",
                  "(string->utf8 (vector-ref model-config 0))",
                  "(vector-ref model-config 1)"):
        assert field in binding
    assert binding.index("(c2-training-state-controls-copy operation controls)") < binding.index(
        "(g3r-joint-original-owner-match")
    wrapper = s[s.index("(define (g3r-joint-c2-original-bound-claim!"):]
    assert wrapper.index("(g3r-joint-p1-provenance-check! p1-ledger operation)") < wrapper.index(
        "(g3r-joint-original-binding-check!") < wrapper.index(
        "(g3r-joint-c2-claim! o2-ledger p1 (vector-ref binding 9))")
    assert "(result (vector binding p1-ledger #f))" in wrapper
    assert "(g3r-joint-p1-provenance-release! p1-ledger)" in wrapper
    # These incompatible accepted mode contracts bar a claimed TR3 trainer
    # snapshot from serving as the same live G3 original-owner control lease.
    assert "(module-mode-internal model) 'eval" in g
    assert "(vector-ref node 1) e3-mode-train" in p


check(native, scheme, g3, p1)
mutations = (
    (native.replace("generator != hold->generator", "generator == hold->generator"), scheme, g3, p1),
    (native.replace("sealed_model != hold->model", "sealed_model == hold->model"), scheme, g3, p1),
    (native.replace("optimizer != hold->optimizer", "optimizer == hold->optimizer"), scheme, g3, p1),
    (native.replace("hold == candidate && hold->live == 1u", "hold == candidate"), scheme, g3, p1),
    (native, scheme.replace("(g3r-save-tokenizer? tokenizer)", "#t"), g3, p1),
    (native, scheme.replace("(vector-ref o2-ledger 8) 'enrolled", "(vector-ref o2-ledger 8) 'dead"), g3, p1),
    (native, scheme.replace("g3r-save-tokenizer-fingerprint))", "\"wrong\"))"), g3, p1),
    (native, scheme.replace("(string->utf8 (vector-ref model-config 0))", "(vector-ref resolved 0)"), g3, p1),
    (native, scheme.replace("(g3r-joint-original-owner-match\n", "(always-zero\n"), g3, p1),
    (native, scheme.replace("(g3r-joint-c2-claim! o2-ledger p1 (vector-ref binding 9))", "#f"), g3, p1),
    (native, scheme.replace("(g3r-joint-p1-provenance-check! p1-ledger operation)", "#f"), g3, p1),
    (native, scheme.replace("(g3r-joint-p1-provenance-release! p1-ledger)", "#f"), g3, p1),
    (native, scheme, g3.replace("(module-mode-internal model) 'eval", "(module-mode-internal model) 'train"), p1),
)
for changed in mutations:
    try:
        check(*changed)
    except (AssertionError, ValueError):
        continue
    raise AssertionError("original binding mutation escaped source gate")

print("joint original binding source gate: pass (13 mutation controls)")
