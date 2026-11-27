#!/usr/bin/env python3
"""Check the private source projection and classify the pinned retirement IR."""
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.c2.test_handler_order import definition, form, lists, need, parse

CONSTRUCTOR = "native/g3c4_p2g2_terminal_carriers_extension.esk"
EXTENSION = "native/g3c4_p2g2_terminal_carrier_retirement_extension.esk"
FIXTURE = "tests/g3c4/p2_g2_terminal_carrier_retirement_test.esk"
CAPTURE = "g3c4-p2g2-terminal-carriers-capture-retirement"
TAIL = "g3c4-p2g2-terminal-carrier-retirement-tail!"
SCOPE = "g3c4-with-p2g2-terminal-carrier-retirement"
FAIL_STOP = "g3c4-p2g2-carrier-retirement-fail-stop"
SCRUB = "g3c4-p2g2-scrub-eight-bytes!"


def check_source():
    roots = {path: parse((ROOT / path).read_text()) for path in (CONSTRUCTOR, EXTENSION, FIXTURE)}
    constructor = list(lists(roots[CONSTRUCTOR]))
    preflight = [node[2] for node in constructor if node[:2] == ["set!", CAPTURE]]
    need(len(preflight) == 1, "one constructor-owned preflight")
    captures = list(lists(preflight[0]))
    for operand in ("(head (vector-ref roots 0))", "(edge (car head))",
                    "(successor (cdr head))", "(live ((cdr edge) retirement-key operation))",
                    "(eq? bundle (car (car head)))", "(m3t-exact-i64? logical-count)",
                    "(or (= logical-count 1) (= logical-count 2))"):
        need(form(operand) in captures, f"preflight lost {operand}")
    need(form("""
        (vector roots head edge successor bundle live
          (vector-ref bundle 0) (vector-ref bundle 1)
          (vector-ref bundle 2) (vector-ref bundle 3)
          (vector-ref bundle 4) (vector-ref bundle 5)
          (vector-ref bundle (if one? 6 7))
          (vector-ref bundle (if one? 7 6)) one?
          (make-bytevector 1 0) 'live 'dead)
    """) in captures, "ticket must carry the original operands in the fixed order")
    need(form("(proof #t operation)") in constructor
         and form("(vector-set! roots 0 remaining)") in constructor,
         "legacy abort must retain authenticated unlink")
    need(all(node[0] not in ("set-car!", "set-cdr!") for node in constructor if node),
         "private root pairs must remain immutable")

    tail = definition(roots, TAIL, EXTENSION)
    tail_forms = list(lists(tail[2:]))
    need(tail[1] == [TAIL, ["capture", ":", "vector"]], "fixed tail ABI")
    allowed = {"let", "let*", "if", "and", "not", "eq?", "vector-ref",
               "vector-set!", "bytevector-u8-ref", "bytevector-u8-set!", FAIL_STOP, SCRUB}
    # (let ...) bindings have list heads and their two-item clauses bind names.
    operators = {node[0] for node in tail_forms if node and isinstance(node[0], str)
                 and node != ["phase", ["vector-ref", "capture", "15"]]
                 and not (len(node) == 2 and isinstance(node[1], list)
                          and node[1][:2] == ["vector-ref", "capture"])}
    need(operators <= allowed, f"unexpected tail operations: {sorted(operators - allowed)}")
    activation = tail[2]
    need(activation[2] == form(f"""
        (if (eq? (bytevector-u8-ref phase 0) 0)
            (bytevector-u8-set! phase 0 1) ({FAIL_STOP} 134))
    """), "one-use transition must precede all retirement writes")
    writes = [node for node in tail_forms if node[:1] == ["vector-set!"]]
    need(writes == parse("""
        (vector-set! live 0 #f)
        (vector-set! bundle 9 (vector-ref capture 17))
        (vector-set! roots 0 successor)
    """), "fixed tombstone and exact successor store")
    scrubs = [node for node in tail_forms if node[:1] in (["bytevector-u8-set!"], [SCRUB])]
    need(scrubs == parse(f"""
        (bytevector-u8-set! phase 0 1)
        (bytevector-u8-set! first-raw 0 0)
        ({SCRUB} first-staging 0 1 2 3 4 5 6 7)
        (bytevector-u8-set! pending-raw 0 0)
        (bytevector-u8-set! pending-raw 1 0)
        ({SCRUB} pending-staging 0 1 2 3 4 5 6 7)
        ({SCRUB} pending-staging 8 9 10 11 12 13 14 15)
        (bytevector-u8-set! second-raw 0 0)
        ({SCRUB} second-staging 0 1 2 3 4 5 6 7)
        (bytevector-u8-set! unused 0 0)
        (bytevector-u8-set! unused 1 0)
    """), "only the six original work carriers and unselected text may be scrubbed")
    need(form("(if (vector-ref capture 14) (bytevector-u8-set! unused 1 0))") in tail_forms,
         "the second unused byte only exists when text one was selected")
    need(form("""(if (not (and (eq? (vector-ref roots 0) head)
                              (eq? (vector-ref live 0) #t)
                              (eq? (vector-ref bundle 9) (vector-ref capture 16))))
                    (g3c4-p2g2-carrier-retirement-fail-stop 134))""") in tail_forms,
         "stale activation must fail-stop before carrier writes")
    need(form(f"(extern void {FAIL_STOP} i64 :real _Exit)") in roots[EXTENSION],
         "use the direct process fail-stop")
    extension_forms = list(lists(roots[EXTENSION]))
    need(form(f"""(#t (if (eq? (bytevector-u8-ref phase 0) 0)
                          (begin (bytevector-u8-set! phase 0 2) (raise caught))
                          ({FAIL_STOP} 134)))""") in extension_forms,
         "preserve the original pre-boundary exception and fail-stop after activation")
    need(form("(if (eq? (bytevector-u8-ref phase 0) 0) (bytevector-u8-set! phase 0 2))")
         in extension_forms, "unconsumed ticket must expire on scope exit")

    references = "|".join(map(re.escape, (CAPTURE, TAIL, SCOPE, Path(EXTENSION).name)))
    files = subprocess.check_output(
        ["git", "grep", "--untracked", "-IlzE", references, "--", "."], cwd=ROOT,
    ).decode().split("\0")
    need(set(files) - {""} == {
        CONSTRUCTOR, EXTENSION, FIXTURE, Path(__file__).relative_to(ROOT).as_posix(),
        "native/g3c4_p2g2_terminal_carrier_retirement_source_closure.txt",
    }, "retirement escaped its private source, fixture or gate projection")
    print("P2/G2 carrier retirement source PASS")


def classify_ir(path):
    name = f'@"{TAIL}"'
    text = path.read_text()
    functions = re.findall(
        rf"^define [^\n]* {re.escape(name)}\([^\n]*\) [^\n]*\{{\n(.*?)^}}",
        text, re.M | re.S,
    )
    need(len(functions) == 1, "expected one emitted static retirement tail")
    expected_blocks = {
        "@_Exit": {"then", "else"},
        "@arena_allocate": {"arena_allocated"},
        "@arena_allocate_tensor_with_header": {"vref_nd_bounds_ok"},
        "@eshkol_region_write_barrier_checked_v1": {"vset_vec_ok"},
        "@eshkol_runtime_emergency_raise_v1": {"arena_allocation_failed", "wb_failure"},
    }
    error_blocks = {
        "bounds_fail", "tvref_bounds_fail", "vref_bounds_fail",
        "vref_nd_bounds_fail", "vset_tensor_fail", "vset_vec_fail",
    }
    expected_blocks["@eshkol_make_exception_with_header"] = error_blocks
    expected_blocks["@eshkol_raise"] = error_blocks
    counts = Counter()
    block = "entry"
    for line in functions[0].splitlines():
        label = re.match(r"^([\w.-]+):", line)
        if label:
            block = re.sub(r"\d+$", "", label[1])
        if not re.search(r"^\s+(?:%[^ ]+ = )?(?:(?:musttail|tail|notail) )?(?:call|invoke|callbr) ", line):
            continue
        target = re.search(r'(@[\w.-]+)\(', line)
        need(target and block in expected_blocks.get(target[1], ()),
             f"unexpected retirement IR call in {block}: {line.strip()}")
        counts[target[1]] += 1
    need(counts["@_Exit"] == 2 and counts["@eshkol_region_write_barrier_checked_v1"] == 3,
         "expected two fail-stop edges and exactly three checked tagged stores")
    print("P2/G2 carrier retirement emitted calls PASS")
    for callee, count in sorted(counts.items()):
        print(f"{callee} {count}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        classify_ir(Path(sys.argv[1]))
    else:
        check_source()
