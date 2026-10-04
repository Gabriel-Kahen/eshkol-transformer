"""Closed private C2/SHA receipt topology and owner handoff checks."""
from pathlib import Path
import hashlib
import unittest

from tests.c2.test_handler_order import (consecutive, definition, form, handler,
                                         lists, parse, protected, render)

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "native/g3r_c2_sha_receipt_extension.esk"
STAGE = ROOT / "native/g3r_c2_sha_stage_extension.esk"
ROOT_FILE = ROOT / "native/g3r_c2_sha_receipt_root.esk"
MANIFEST = ROOT / "native/g3r_c2_sha_receipt_source_closure.txt"
STAGE_MANIFEST = ROOT / "native/g3r_c2_sha_stage_source_closure.txt"
RUNNER = ROOT / "scripts/test-g3r-c2-sha-receipt.sh"
FIXTURE = ROOT / "tests/g3r/c2_sha_receipt_test.esk"
PINNED = {
    "native/c2_checkpoint_load_extension.esk":
        "adb2f3f832f9f91ec65a665ad2ccfe2665ce56463a209978794ae34df659770d",
    "native/c2_public_extension.esk":
        "48641732a31964d18ab3f82689ba9a1ce1d563c1152ef98b19b94f2b49ee297e",
    "native/k2_wave2_extension.esk":
        "cc501d9417cd4387b8d16f75fc864d2ab1357185b312751f727895b811b5a950",
    "native/g3r_c2_sha_stage_root.esk":
        "4678059d58eafbd9a0358dd1f9fb97704a6e2c3a07ecfd53d95b6889aaae35c6",
    "scripts/test-g3r-c2-sha-stage.sh":
        "dcffd1648f2a8eaebb9392e4aee7033f62eccef7c9f993e4765a1b619f277387",
}
LOADS = [
    "d2_wave2_root.esk", "c2_d2_cursor_pair_extension.esk",
    "o2_wave2_extension.esk", "k2_wave2_extension.esk",
    "c2_o2_reconstruct_extension.esk", "c2_x1_canonical_extension.esk",
    "c2_training_state_extension.esk", "c2_persistence_policy_extension.esk",
    "c2_checkpoint_load_extension.esk", "g3r_c2_sha_stage_extension.esk",
    "g3r_c2_sha_receipt_extension.esk",
]


def verify_root_and_closure(root: str, closure: str, predecessor: str) -> None:
    forms = parse(root)
    loads = [form[1].strip('"') for form in forms
             if isinstance(form, list) and len(form) == 2 and form[0] == "load"]
    if len(forms) != len(LOADS) or loads != LOADS:
        raise ValueError("private K2/C2 receipt load order differs")
    expected = predecessor.splitlines()
    expected[0] = "native/g3r_c2_sha_receipt_root.esk"
    expected.insert(expected.index("native/o2_wave2_extension.esk") + 1,
                    "native/k2_wave2_extension.esk")
    expected.append("native/g3r_c2_sha_receipt_extension.esk")
    if closure.splitlines() != expected or len(set(expected)) != len(expected):
        raise ValueError("private receipt source closure differs")


def verify_handoff(source: str) -> None:
    forms = parse(source)
    load = definition({"receipt": forms}, "g3r-c2-load-with-sha-internal")
    if "(c2-runtime-reserve-exception-handlers 14)" not in source:
        raise ValueError("receipt lacks fourteen-frame handler reserve")
    stage_guards = protected(
        load, "(set! stage (g3r-c2-sha-stage-internal path policy))", 3,
        "(g3r-c2-receipt-retire-stage! stage)")
    owner_guards = protected(
        load,
        "(set! owner (c2-checkpoint-load-reconstruct-internal stage 'checkpoint-load))",
        3, "(g3r-c2-receipt-release-owner! owner #f release-record defect)")
    for guards in (stage_guards, owner_guards):
        if "(g3r-c2-sha-fail-stop 134)" not in render(handler(guards[1])):
            raise ValueError("returned authority lacks cleanup fail-stop")
    consecutive(load,
                "(set! stage (g3r-c2-sha-stage-internal path policy))",
                "(g3r-c2-receipt-after-stage-return-internal stage)",
                "(vector-set! stage-cell 0 stage)")
    consecutive(load,
                "(set! owner (c2-checkpoint-load-reconstruct-internal stage 'checkpoint-load))",
                "(g3r-c2-receipt-after-owner-return-internal owner)",
                "(vector-set! owner-cell 0 owner)")
    if ("(let ((stage #f) (stage-cut (vector #f #f)))" not in source
            or "(let ((owner #f))" not in source
            or "(release-record (c2-training-state-release-record #f))" not in source):
        raise ValueError("returned authority lacks preallocated lexical slots")
    protected(load,
              "(c2-checkpoint-load-reconstruct-internal stage 'checkpoint-load)",
              3, "(g3r-c2-receipt-release-owner! owner #f release-record defect)")
    if "c2-public-checkpoint-load" in source or "c2-checkpoint-load-stage-internal" in source:
        raise ValueError("receipt reopens C2 via another LOAD")
    if "(g3r-c2-receipt-cleanup-failure!" not in source:
        raise ValueError("receipt lacks outer cleanup")


def verify_runner(source: str) -> None:
    required = (
        "test-g3r-c2-sha-stage.sh", "-DET_G3R_C2_STAGE_SHA_PRIVATE",
        "-DET_K2_TESTING", "c2_sha_receipt_test_bridge.c",
        "g3r_c2_sha_receipt_source_closure.txt",
        "build_mode normal", "build_mode repeat", "build_mode sanitize",
        "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1",
        "UBSAN_OPTIONS=halt_on_error=1",
        "LSAN_OPTIONS=exitcode=23:report_objects=1",
        "--dump-ir", "--emit-depfile", "receipt-test.ll", "receipt-test.o",
        "--emit-object", "stage-dep.o", "object-build.stdout",
        "object-build.stderr", "dep-object-undefined-symbols.txt",
        "dep-object-defined-symbols.txt", "ir-object-build.stdout",
        "ir-object-build.stderr", "unexpected LLVM IR object compiler diagnostics",
        "native-undefined-symbols.txt", "eshkol-object-undefined-symbols.txt",
        "linked-symbols.txt", "source-closure.txt",
    )
    if any(item not in source for item in required):
        raise ValueError("compiled three-mode receipt gate is incomplete")
    if (source.count("local -a eshkol_inputs=(") != 1
            or source.count('"${eshkol_inputs[@]}"') != 2
            or source.count('--emit-depfile "$directory/stage.d"') != 1
            or not (source.index("--dump-ir") < source.index("--emit-object")
                    < source.index('--emit-depfile "$directory/stage.d"'))
            or 'test ! -s "$directory/object-build.stderr"' not in source
            or 'test -s "$directory/stage-dep.o"' not in source
            or 'if diagnostics != expected:' not in source):
        raise ValueError("receipt object depfile must use the executable inputs")


def verify_fault_bindings(source: str) -> None:
    forms = parse(source)
    for name, parameters in (
            ("g3r-c2-receipt-identity", []),
            ("g3r-c2-receipt-after-owner-enrolled-internal", []),
            ("g3r-c2-receipt-after-stage-return-internal", ["stage"]),
            ("g3r-c2-receipt-after-owner-return-internal", ["owner"])):
        matches = [form for form in forms if isinstance(form, list)
                   and len(form) == 3 and form[:2] == ["define", name]]
        if (len(matches) != 1 or not isinstance(matches[0][2], list)
                or matches[0][2][:2] != ["lambda", parameters]):
            raise ValueError("receipt fault helper needs a mutable binding")


def verify_race_assertions(source: str) -> None:
    forms = parse(source)
    phases = (
        (("same second image publishes one exact receipt", "record"),
         ("same second image publishes one live C2 owner",
          "(= (vector-ref counts-after-publication 0) (+ owners-before 1))"),
         ("same second image preserves prior busy C2 count",
          "(= (vector-ref counts-after-publication 1) busy-before)"),
         ("same second image publishes one receipt identity",
          "(= (receipt-count) (+ before 1))"),
         ("new receipt is live before borrowing",
          "(eq? (vector-ref record 1) 'live)")),
        (("borrowed C2 owner no longer counts live",
          "(= (vector-ref counts-during-borrow 0) owners-before)"),
         ("borrowed C2 owner counts busy exactly once",
          "(= (vector-ref counts-during-borrow 1) (+ busy-before 1))"),
         ("receipt records the active borrow identity",
          "(eq? (vector-ref record 1) borrowed)"),
         ("same second image B supplies the stored receipt SHA",
          "(bytes=? sha-copy expected-b)"),
         ("late path replacement leaves image A at the mutable path",
          "(bytes=? expected-a (c1-sha256 (c1-read-file mutable-path "
          "(c2-policy-c1-subpolicy-internal 'checkpoint-load policy) "
          "'checkpoint-load)))"),
         ("same second image B supplies the Philox controls",
          "(equal? (vector-ref controls 0) "
          "(vector 'philox4x32-10 1 1729 -1 -9223372036854775808))"),
         ("consumed stage sidecars are dead with zero SHA",
          "(all-sidecars-dead?)"),
         ("receipt load leaves native image reader and FD counts flat",
          "(native-flat?)")),
        (("borrow end restores one live C2 owner",
          "(= (vector-ref counts-after-borrow-end 0) (+ owners-before 1))"),
         ("borrow end restores prior busy C2 count",
          "(= (vector-ref counts-after-borrow-end 1) busy-before)"),
         ("borrow end restores live receipt status",
          "(eq? (vector-ref record 1) 'live)")),
    )
    checks = {}
    for index, node in enumerate(forms):
        if isinstance(node, list) and node[:1] == ["check"] and len(node) == 3:
            checks.setdefault(node[1], []).append((index, node[2]))
    def position(head: str, name: str) -> int:
        matches = [i for i, node in enumerate(forms) if isinstance(node, list)
                   and node[:2] == [head, name]]
        if len(matches) != 1:
            raise ValueError("receipt race phase anchor differs")
        return matches[0]
    boundaries = (position("define", "record"),
                  position("define", "borrowed"),
                  position("check", '"borrow end returns live receipt"'),
                  position("check", '"receipt release retires exactly one owner"'))
    if boundaries != tuple(sorted(boundaries)):
        raise ValueError("receipt race phase order differs")
    for phase, (start, end) in zip(phases, zip(boundaries, boundaries[1:])):
        for label, expected in phase:
            match = checks.get(f'"{label}"', [])
            if len(match) != 1 or match[0][1] != form(expected):
                raise ValueError(f"receipt race check differs: {label}")
            if not start < match[0][0] < end:
                raise ValueError(f"receipt race phase order differs: {label}")


class C2ShaReceiptStaticTest(unittest.TestCase):
    def test_private_sources_and_handoff(self) -> None:
        for relative, digest in PINNED.items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)
        verify_root_and_closure(ROOT_FILE.read_text(), MANIFEST.read_text(),
                                STAGE_MANIFEST.read_text())
        verify_handoff(RECEIPT.read_text())
        verify_fault_bindings(RECEIPT.read_text())
        verify_runner(RUNNER.read_text())
        parse(STAGE.read_text())
        verify_race_assertions(FIXTURE.read_text())
        self.assertIn("g3r-c2-sha-stage-finalize-consumed-internal!", STAGE.read_text())

    def test_near_misses_reject(self) -> None:
        root = ROOT_FILE.read_text()
        closure = MANIFEST.read_text()
        predecessor = STAGE_MANIFEST.read_text()
        source = RECEIPT.read_text()
        runner = RUNNER.read_text()
        with self.assertRaisesRegex(ValueError, "load order"):
            verify_root_and_closure(root.replace(
                '(load "k2_wave2_extension.esk")',
                '(load "c2_public_extension.esk")'), closure, predecessor)
        with self.assertRaisesRegex(ValueError, "closure"):
            verify_root_and_closure(root, closure.replace(
                "native/g3r_c2_sha_receipt_extension.esk",
                "native/c2_public_extension.esk"), predecessor)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            verify_handoff(source.replace(
                "(vector-set! owner-cell 0 owner)",
                "(c2-load-exact-live-owner? owner)\n              (vector-set! owner-cell 0 owner)"))
        with self.assertRaisesRegex(ValueError, "guard depth"):
            verify_handoff(source.replace(
                "(guard (cleanup-defect (#t (g3r-c2-sha-fail-stop 134)))",
                "(begin", 1))
        with self.assertRaisesRegex(ValueError, "guard depth"):
            verify_handoff(source.replace(
                "(guard (cleanup-defect (#t (g3r-c2-sha-fail-stop 134)))",
                "(begin", 1).replace(
                "(guard (cleanup-defect (#t (g3r-c2-sha-fail-stop 134)))",
                "(begin", 1))
        with self.assertRaisesRegex(ValueError, "three-mode"):
            verify_runner(runner.replace("build_mode sanitize", "build_mode normal"))
        with self.assertRaisesRegex(ValueError, "object depfile"):
            verify_runner(runner.replace('"${eshkol_inputs[@]}"',
                                         '"${eshkol_inputs[@]:1}"', 1))
        with self.assertRaisesRegex(ValueError, "three-mode"):
            verify_runner(runner.replace('--emit-object \\\n', '', 1))
        with self.assertRaisesRegex(ValueError, "object depfile"):
            verify_runner(runner.replace('if diagnostics != expected:',
                                         'if False:', 1))
        with self.assertRaisesRegex(ValueError, "mutable binding"):
            verify_fault_bindings(source.replace(
                "(define g3r-c2-receipt-identity\n  (lambda ()",
                "(define (g3r-c2-receipt-identity)\n  (begin", 1))

    def test_race_assertion_near_misses_reject(self) -> None:
        source = FIXTURE.read_text()
        with self.assertRaisesRegex(ValueError, "borrowed C2 owner no longer"):
            verify_race_assertions(source.replace(
                "(= (vector-ref counts-during-borrow 0) owners-before)",
                "(= (vector-ref counts-during-borrow 0) (+ owners-before 1))"))
        with self.assertRaisesRegex(ValueError, "stored receipt SHA"):
            verify_race_assertions(source.replace(
                "(bytes=? sha-copy expected-b)", "#t", 1))
        late = ('(check "borrow end restores one live C2 owner"\n'
                '  (= (vector-ref counts-after-borrow-end 0) (+ owners-before 1)))')
        with self.assertRaisesRegex(ValueError, "receipt race phase order"):
            verify_race_assertions(source.replace(late, "").replace(
                "(define borrowed\n", late + "\n(define borrowed\n", 1))


if __name__ == "__main__":
    unittest.main()
