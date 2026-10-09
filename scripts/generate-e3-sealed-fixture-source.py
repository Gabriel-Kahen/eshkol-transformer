#!/usr/bin/env python3
"""Generate a noninstalled, feature-private P1/I2/M3 fixture source overlay.

This is a source staging tool. It never changes the canonical production files.
The generated tuple still needs separately reviewed FFI/object/source inventories
and a compiled normal/repeat/sanitizer gate before it is runnable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "p1": ("internal/p1/lib/transformer/module.esk",
           "1146eee9e053b4a761c24f56b75ed43e756ec2d907d0de48b04e72abe805775d"),
    "i2": ("native/i2_wave2_extension.esk",
           "66bdd604a8bffcf90913961084993b2410bd89d3dbb7a9aed0de0524115164dc"),
    "m3t": ("native/m3t_transport_extension.esk",
            "71247e722d72764ce1b3400cb3f818ba12c3754dd11a6c742ffcec63942d5815"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected one source anchor: {old[:90]!r}")
    return source.replace(old, new, 1)


def pinned(kind: str) -> str:
    path, expected = INPUTS[kind]
    data = (ROOT / path).read_bytes()
    if sha(data) != expected:
        raise ValueError(f"{kind} predecessor changed")
    return data.decode("utf-8")


def p1() -> bytes:
    source = pinned("p1")
    source = once(source,
        "(extern i64 p1-native-construction-begin ptr\n"
        "        :real et_p1_private_construction_begin_v1)",
        "(extern i64 p1-native-construction-begin ptr\n"
        "        :real et_p1_private_construction_begin_v1)\n"
        "(extern i64 p1-native-candidate-construction-begin ptr\n"
        "        :real et_p1_private_candidate_construction_begin_v1)\n"
        "(extern i64 p1-native-candidate-graph-preflight ptr ptr\n"
        "        :real et_p1_private_candidate_graph_preflight_v1)\n"
        "(extern i64 p1-native-candidate-graph-revoke ptr ptr\n"
        "        :real et_p1_private_candidate_graph_revoke_v1)\n"
        "(extern void p1-candidate-invariant-fail :real abort)")
    source = once(source,
        "         module-construction-parameters-internal\n         p1-trusted-surface)",
        "         module-construction-parameters-internal\n"
        "         module-candidate-construction-begin-internal\n"
        "         module-candidate-construction-preflight-internal!\n"
        "         module-candidate-construction-revoke-internal!\n"
        "         p1-trusted-surface)")
    source = once(source, "(define (construction-begin provider-name)",
                  "(define (construction-begin-raw provider-name candidate?)")
    source = once(source,
        "(p1-native-construction-begin native-context)",
        "(if candidate?\n"
        "                          (p1-native-candidate-construction-begin native-context)\n"
        "                          (p1-native-construction-begin native-context))")
    source = once(source, "(define (construction-parameters identity)",
        "(define (construction-begin provider-name)\n"
        "  (construction-begin-raw provider-name #f))\n"
        "(define (g3r-construction-begin provider-name)\n"
        "  (construction-begin-raw provider-name #t))\n"
        "(define (construction-parameters identity)")
    fragment = (ROOT / "tests/e3_private/sealed_fixture_p1.esk.inc").read_text()
    source = once(source, "(define p1-trusted-surface\n", fragment + "\n(define p1-trusted-surface\n")
    source = once(source,
        "  (lambda (identity)\n    (construction-seal-prepared! identity))\n  )))",
        "  (lambda (identity)\n    (construction-seal-prepared! identity))\n"
        "  ;; Append-only zero-fixture private slots 73..75.\n"
        "  (lambda (provider-name) (g3r-construction-begin provider-name))\n"
        "  (lambda (identity root handles schedule)\n"
        "    (g3r-construction-preflight identity root handles schedule))\n"
        "  (lambda (identity) (g3r-construction-revoke! identity))\n  )))")
    source += "\n" + """(define (module-candidate-construction-begin-internal provider-name)
  ((vector-ref p1-trusted-surface 73) provider-name))
(define (module-candidate-construction-preflight-internal! identity root handles schedule)
  ((vector-ref p1-trusted-surface 74) identity root handles schedule))
(define (module-candidate-construction-revoke-internal! identity)
  ((vector-ref p1-trusted-surface 75) identity))
"""
    return source.replace("(e3-native-invariant-fail)",
                          "(p1-candidate-invariant-fail)").encode()


def i2() -> bytes:
    source = pinned("i2")
    source = once(source, "((eq? route 'm3t)\n",
                  "((or (eq? route 'm3t) (eq? route 'm3t-candidate))\n")
    source = once(source, "(define (i2-construction-begin-internal)",
                  "(define (i2-construction-begin-raw candidate?)")
    source = once(source,
        "'m3t #f))\n         (next (cons record",
        "(if candidate? 'm3t-candidate 'm3t) #f))\n         (next (cons record")
    source = once(source,
        "(module-construction-begin-internal i2-provider-name)",
        "(if candidate?\n"
        "                                  (module-candidate-construction-begin-internal i2-provider-name)\n"
        "                                  (module-construction-begin-internal i2-provider-name))")
    source = once(source, "(define (i2-construction-seal-internal! value)",
        "(define (i2-construction-begin-internal)\n"
        "  (i2-construction-begin-raw #f))\n"
        "(define (i2-candidate-construction-begin-internal)\n"
        "  (i2-construction-begin-raw #t))\n"
        "(define (i2-construction-seal-internal! value)")
    source = once(source,
        "(define (i2-construction-abort-internal! value)",
        (ROOT / "tests/e3_private/sealed_fixture_i2.esk.inc").read_text() +
        "\n(define (i2-construction-abort-internal! value)")
    source = once(source,
        "(if (not (eq? (vector-ref record 9) 'm3t))",
        "(if (not (or (eq? (vector-ref record 9) 'm3t)\n"
        "                 (eq? (vector-ref record 9) 'm3t-candidate)))")
    source = once(source,
        "value 'm3t 'i2-construction-abort-internal!)",
        "value (vector-ref record 9) 'i2-construction-abort-internal!)")
    return source.encode()


def m3t() -> bytes:
    canonical = pinned("m3t").encode()
    wrong_tie = ROOT / "scripts/generate-e3-zero-wrong-tie-source.py"
    if sha(wrong_tie.read_bytes()) != "995a5460e4141976322510c0ed4f3704e615a119d1d7d6c0f29b34d2b3a4c361":
        raise ValueError("accepted wrong-tie producer changed")
    spec = importlib.util.spec_from_file_location("accepted_wrong_tie", wrong_tie)
    if spec is None or spec.loader is None:
        raise ValueError("accepted wrong-tie producer is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.compose(canonical).decode()
    externs = """(extern i64 m3t-native-candidate-binding-preflight ptr i64 ptr ptr
        :real et_m3t_private_candidate_binding_preflight_v1)
(extern i64 m3t-native-candidate-initializer-retire-preflight ptr
        :real et_m3t_private_candidate_initializer_retire_preflight_v1)
(extern i64 m3t-native-candidate-retire-preflight ptr ptr ptr
        :real et_m3t_private_candidate_retire_preflight_v1)
(extern i64 m3t-native-candidate-retire-commit ptr
        :real et_m3t_private_candidate_retire_commit_v1)
(extern i64 m3t-native-candidate-initializer-retire-commit ptr
        :real et_m3t_private_candidate_initializer_retire_commit_v1)
(extern void m3t-candidate-invariant-fail :real abort)

"""
    source = once(source, "(define m3t-profile ", externs + "(define m3t-profile ")
    start = source.index("(define (m3t-public-model-create config state)")
    end = source.index("(define (m3t-public-model-profile model)", start)
    constructor = source[start:end]
    constructor = once(constructor,
        "(define (m3t-public-model-create config state)",
        "(define (m3t-model-create-raw config state candidate?)")
    constructor = once(constructor, "(cleanup (vector #f #f)))",
        "(cleanup (vector #f #f))\n"
        "         (ledger (if candidate?\n"
        "                   (vector 'pending entry initial successor #f #f #f\n"
        "                           (make-vector 14 #f) #f #f #f) #f)))")
    constructor = once(constructor,
        "(vector-set! entry 11 m3t-profile)",
        "(vector-set! entry 11 m3t-profile)\n"
        "    (if candidate?\n"
        "        (vector-set! m3t-candidate-root 0\n"
        "          (cons ledger (vector-ref m3t-candidate-root 0))))")
    constructor = once(constructor,
        "(begin (vector-set! entry 2 'dead) (vector-set! successor 2 'dead)\n"
        "                 (m3t-native-fail operation))",
        "(begin (vector-set! entry 2 'dead) (vector-set! successor 2 'dead)\n"
        "                 (if candidate? (vector-set! ledger 0 'dead))\n"
        "                 (m3t-native-fail operation))")
    constructor = once(constructor,
        "(vector-set! entry 3 native)",
        "(vector-set! entry 3 native)\n"
        "      (if candidate? (vector-set! ledger 4 native))")
    constructor = once(constructor,
        """(if (not (vector-ref cleanup 1))
                       (begin
                         (if (vector-ref cleanup 0)
                             (i2-construction-abort-internal! (vector-ref cleanup 0)))
                         (m3t-check (m3t-native-owner-abort native) operation)
                         (vector-set! entry 2 'dead)
                         (vector-set! successor 2 'dead)))""",
        """(if (not (vector-ref cleanup 1))
                       (guard (cleanup-failure
                                (#t (m3t-candidate-invariant-fail)))
                         (if (vector-ref cleanup 0)
                             (i2-construction-abort-internal! (vector-ref cleanup 0)))
                         (m3t-check (m3t-native-owner-abort native) operation)
                         (vector-set! entry 2 'dead)
                         (vector-set! successor 2 'dead)
                         (if candidate? (vector-set! ledger 0 'dead))))""")
    constructor = once(constructor,
        "(i2-construction-begin-internal)))",
        "(if candidate? (i2-candidate-construction-begin-internal)\n"
        "                           (i2-construction-begin-internal))))")
    constructor = once(constructor,
        "(vector-set! cleanup 0 construction)",
        "(vector-set! cleanup 0 construction)\n"
        "            (if candidate?\n"
        "                (begin (vector-set! ledger 5 (vector-ref construction 1))\n"
        "                       (vector-set! ledger 6 construction)))")
    constructor = once(constructor,
        "(vector-set! entry 0 root)",
        "(vector-set! entry 0 root)\n"
        "            (if candidate? (vector-set! ledger 10 root))")
    constructor = once(constructor,
        "(vector-set! (vector-ref entry 8) index handle)",
        "(vector-set! (vector-ref entry 8) index handle)\n"
        "                      (if candidate?\n"
        "                          (vector-set! (vector-ref ledger 7) index\n"
        "                                       (vector parameter handle\n"
        "                                         (parameter-handle-tensor-identity-internal handle))))")
    constructor = once(constructor,
        "(m3t-validate-schedule schedule (vector-ref entry 8) operation)",
        "(m3t-validate-schedule schedule (vector-ref entry 8) operation)\n"
        "              (if candidate? (vector-set! ledger 9 schedule))")
    constructor = once(constructor,
        "(i2-construction-seal-internal! construction)",
        "(if candidate? (i2-candidate-construction-seal-internal! construction)\n"
        "                           (i2-construction-seal-internal! construction))")
    constructor = once(constructor,
        "(vector-set! entry 2 'live)\n            root",
        "(vector-set! entry 2 'live)\n"
        "            (if candidate? (vector-set! ledger 0 'sealed))\n"
        "            root")
    constructor += "(define (m3t-public-model-create config state)\n" \
                   "  (m3t-model-create-raw config state #f))\n" \
                   "(define (m3t-candidate-model-create config state)\n" \
                   "  (m3t-model-create-raw config state #t))\n\n"
    fragment = (ROOT / "tests/e3_private/sealed_fixture_m3t.esk.inc").read_text()
    source = source[:start] + constructor + fragment + "\n" + source[end:]
    # The candidate-specific tail never relies on the later E3 source load.
    return source.replace("(e3-native-invariant-fail)",
                          "(m3t-candidate-invariant-fail)").encode()


def outputs() -> dict[str, bytes]:
    root_bytes = (ROOT / "native/i2_wave2_root.esk").read_bytes()
    if sha(root_bytes) != "f1314ccd4c5a382733115005c93d8712290bd63c5373eac2509093989fefad33":
        raise ValueError("I2 root changed")
    prefix = root_bytes.decode().split(
        "(extern i64 i2-native-construction-parameter-preflight", 1)[0]
    if not prefix.startswith(";;; Canonical Wave 2 I2/P1 trusted compilation root."):
        raise ValueError("I2 root prefix changed")
    return {
        "transformer/module.esk": p1(),
        "i2_wave2_root.esk": prefix.encode() + i2(),
        "m3t_transport_extension.esk": m3t(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = outputs()
    if not args.check:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError("fresh output directory required")
        args.output_dir.mkdir(parents=True)
    for name, data in expected.items():
        target = args.output_dir / name
        if args.check:
            if target.read_bytes() != data:
                raise ValueError(f"generated source differs: {name}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        print(f"{sha(data)}  {name}")


if __name__ == "__main__":
    main()
