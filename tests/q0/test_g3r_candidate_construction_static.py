"""No toolchain needed: private overlay and ordinary-source isolation checks."""

from pathlib import Path
import hashlib
import re
import subprocess
import sys
import tempfile
import unittest
from tests.g3r.check_candidate_construction_inputs import (
    check_sanitizer_generated, compare_depfile, depfile_paths, load_allowlist,
    require_regular,
)


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/build-g3r-candidate-p1-overlay.py"
MODULE = ROOT / "internal/p1/lib/transformer/module.esk"
PRIVATE = ROOT / "native/g3r_candidate_construction_extension.esk"
ROOT_SOURCE = ROOT / "native/g3r_candidate_construction_root.esk"
ORIGINAL_MODULE_SHA = "1146eee9e053b4a761c24f56b75ed43e756ec2d907d0de48b04e72abe805775d"
UNCHANGED = {
    "native/m3t_package_root.esk": "f3f804cedff58b87c0ba08d3a30a3559bd2d0850d53558a7b3a21b4fd7e01c98",
    "native/i2_wave2_root.esk": "f1314ccd4c5a382733115005c93d8712290bd63c5373eac2509093989fefad33",
    "native/m3t_transport_extension.esk": "71247e722d72764ce1b3400cb3f818ba12c3754dd11a6c742ffcec63942d5815",
    "native/p1_identity.c": "155b2fe2fb6b26c6cd089f56390659cf3790265263e0d34f76b8dc06c8cae90f",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CandidateConstructionStatic(unittest.TestCase):
    def test_generated_asan_receipt_and_reporter_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            mode = evidence / "sanitize"
            mode.mkdir()
            raw = mode / "candidate.raw.o.ll"
            derived = mode / "candidate.asan.ll"
            receipt = mode / "candidate.asan.json"
            raw.write_bytes(b"define i32 @read(ptr %p) {\nentry:\n"
                            b" %v = load i32, ptr %p\n ret i32 %v\n}\n")
            invoked = subprocess.run(
                [sys.executable, "-B", str(ROOT / "tests/shared/prepare_eshkol_asan_ir.py"),
                 str(raw), str(derived), str(receipt)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual((invoked.returncode, invoked.stdout, invoked.stderr),
                             (0, "", ""))
            (mode / "candidate.asan.o").write_bytes(b"retained object")
            symbols = evidence / "sanitize.asan-symbols.stdout"
            for step in ("asan-prepare", "asan-compile", "asan-symbols"):
                for suffix, value in (("stdout", ""), ("stderr", ""),
                                      ("exit", "0\n")):
                    (evidence / f"sanitize.{step}.{suffix}").write_text(value)
            symbols.write_text("__asan_report_load4 U         \n")
            check_sanitizer_generated(ROOT, evidence)
            original = derived.read_bytes()
            derived.write_bytes(original.replace(b"@read", b"@evil", 1))
            with self.assertRaisesRegex(ValueError, "derived IR differs"):
                check_sanitizer_generated(ROOT, evidence)
            derived.write_bytes(original)
            original_receipt = receipt.read_bytes()
            receipt.write_bytes(original_receipt + b" ")
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                check_sanitizer_generated(ROOT, evidence)
            receipt.write_bytes(original_receipt)
            symbols.write_text("__asan_init U\n")
            with self.assertRaisesRegex(ValueError, "lacks ASan"):
                check_sanitizer_generated(ROOT, evidence)
            symbols.write_text("__asan_report_load4 U         \n")
            (evidence / "sanitize.asan-compile.exit").write_text("137\n")
            with self.assertRaisesRegex(ValueError, "numeric exit"):
                check_sanitizer_generated(ROOT, evidence)
            (evidence / "sanitize.asan-compile.exit").write_text("0\n")
            (evidence / "sanitize.asan-prepare.stderr").write_text("unexpected\n")
            with self.assertRaisesRegex(ValueError, "unexpected stderr"):
                check_sanitizer_generated(ROOT, evidence)
            (evidence / "sanitize.asan-prepare.stderr").write_text("")
            symbols.unlink()
            symbols.symlink_to(evidence / "sanitize.asan-prepare.stdout")
            with self.assertRaisesRegex(ValueError, "nonregular or symlink"):
                check_sanitizer_generated(ROOT, evidence)

    def test_runner_uses_approved_generated_asan_recipe(self) -> None:
        runner = (ROOT / "scripts/test-g3r-candidate-construction.sh").read_text()
        self.assertIn("tests/shared/prepare_eshkol_asan_ir.py", runner)
        self.assertIn('"$evidence/$mode/candidate.asan.ll"', runner)
        self.assertIn('"$evidence/$mode/candidate.asan.json"', runner)
        self.assertIn('"$evidence/$mode/candidate.asan.o"', runner)
        self.assertNotIn("candidate.instrumented.o", runner)

    def test_depfile_and_evidence_file_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = root / "entry.esk"
            private = root / "private.esk"
            ordinary = root / "ordinary.esk"
            native = root / "native.c"
            extra = root / "extra.esk"
            for dependency in (entry, private, ordinary, native, extra):
                dependency.write_text("fixed source\n")
            depfile = root / "input.d"
            target = root / "output.o"
            expected = {entry, private, native}
            def write(target_name: Path, inputs: tuple[Path, ...]) -> None:
                depfile.write_text(f"{target_name}: {' '.join(map(str, inputs))}\n")
            write(target, (entry, private, native))
            self.assertEqual(set(compare_depfile(depfile, target, expected)), expected)
            write(root / "wrong-object.o", (entry, private, native))
            with self.assertRaises(ValueError):
                depfile_paths(depfile, target)
            for inputs in ((entry, ordinary, native), (private, native),
                           (entry, private), (entry, private, native, extra)):
                write(target, inputs)
                with self.assertRaises(ValueError):
                    compare_depfile(depfile, target, expected)
            write(target, (Path("../entry.esk"),))
            with self.assertRaises(ValueError):
                depfile_paths(depfile, target)
            alias = root / "alias"
            alias.symlink_to(entry)
            with self.assertRaises(ValueError):
                require_regular(alias)

    def test_depfile_allowlist_is_immutable(self) -> None:
        original = ROOT / "tests/g3r/candidate_depfile_allowlist.json"
        self.assertEqual(load_allowlist(original)["version"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "allowlist.json"
            changed.write_bytes(original.read_bytes() + b" \n")
            with self.assertRaises(ValueError):
                load_allowlist(changed)

    def test_candidate_and_ordinary_load_topology(self) -> None:
        allowlist = load_allowlist(ROOT / "tests/g3r/candidate_depfile_allowlist.json")
        p1 = "internal/p1/lib/transformer/module.esk"
        accepted_i2 = set((ROOT / "native/i2_wave2_source_closure.txt").read_text().splitlines())
        self.assertEqual(set(allowlist["eshkol_common"]), accepted_i2 - {p1})
        self.assertEqual(set(allowlist["eshkol_candidate_extra"]), {
            "tests/g3r/candidate_construction_test.esk",
            "native/g3r_candidate_construction_root.esk",
            "native/m3t_package_root.esk",
            "native/m3t_transport_extension.esk",
        })
        self.assertEqual(allowlist["eshkol_off_extra"],
                         ["tests/p1/i2_construction_test.esk"])
        topology = {
            "tests/g3r/candidate_construction_test.esk":
                ['(load "g3r_candidate_construction_root.esk")'],
            "tests/p1/i2_construction_test.esk": ['(load "i2_wave2_root.esk")'],
            "native/g3r_candidate_construction_root.esk":
                ['(load "m3t_package_root.esk")'],
            "native/m3t_package_root.esk":
                ['(load "i2_wave2_root.esk")',
                 '(load "m3t_transport_extension.esk")'],
            "native/i2_wave2_root.esk":
                ['(load "t1_wave1_root.esk")',
                 '(require transformer.checkpoint_internal)'],
            "native/t1_wave1_root.esk":
                ['(require e1b_error_consumer_private)',
                 '(require x1_config_private)', '(require transformer.module)',
                 '(require transformer.persistence_policy_internal)',
                 '(load "eshkol_transformer/token_shard.esk")',
                 '(load "transformer/tokenizer_internal.esk")'],
        }
        for relative, expected in topology.items():
            with self.subTest(relative=relative):
                actual = [line for line in (ROOT / relative).read_text().splitlines()
                          if line.startswith("(load ") or line.startswith("(require ")]
                self.assertEqual(actual, expected)

    def test_ordinary_sources_are_exactly_unchanged(self) -> None:
        self.assertEqual(digest(MODULE), ORIGINAL_MODULE_SHA)
        for relative, expected in UNCHANGED.items():
            with self.subTest(relative=relative):
                self.assertEqual(digest(ROOT / relative), expected)
        self.assertNotIn("candidate-construction-begin", MODULE.read_text())

    def test_private_overlay_is_deterministic_and_noninstalled(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for destination in (first, second):
                subprocess.run(
                    ["python3", str(GENERATOR), destination],
                    cwd=ROOT, check=True, capture_output=True, text=True,
                )
            left = Path(first) / "transformer/module.esk"
            right = Path(second) / "transformer/module.esk"
            self.assertEqual(left.read_bytes(), right.read_bytes())
            source = left.read_text()
            self.assertEqual(source.count("et_p1_private_candidate_construction_begin_v1"), 1)
            self.assertEqual(source.count("et_p1_private_candidate_graph_preflight_v1"), 1)
            self.assertIn("(vector-ref p1-trusted-surface 73)", source)
            self.assertIn("(vector-ref p1-trusted-surface 74)", source)
            for slot in range(75, 81):
                self.assertIn(f"(vector-ref p1-trusted-surface {slot})", source)
            for slot in range(73, 81):
                self.assertEqual(source.count(f"(vector-ref p1-trusted-surface {slot})"), 1)
                self.assertNotIn(f"((vector-ref p1-trusted-surface {slot})", source)
                self.assertIn(f"(let ((fixed (vector-ref p1-trusted-surface {slot})))", source)
            self.assertIn("(define g3r-candidate-ledgers (vector '() #f))", source)
            self.assertEqual(source.count("(define p1-trusted-surface"), 1)
            self.assertLess(source.index("(define g3r-candidate-ledgers"),
                            source.index("(vector\n  (lambda (module)"))
            self.assertNotIn("(define g3r-candidate-ledgers", source[source.index(";; Candidate-only wrapper"):])
            self.assertIn("(candidate-construction-shells identity)", source)
            self.assertIn("(if irreversible? (exit 134)", source)
        self.assertEqual(digest(MODULE), ORIGINAL_MODULE_SHA)

    def test_closed_private_root_and_enrollment_order(self) -> None:
        root = ROOT_SOURCE.read_text()
        self.assertEqual(
            [line for line in root.splitlines() if line.startswith("(load ")],
            ['(load "m3t_package_root.esk")'],
        )
        source = PRIVATE.read_text()
        self.assertNotIn("(provide ", source)
        for name in (
            "g3r-candidate-ledger-begin-internal!",
            "g3r-candidate-model-create-internal!",
            "i2-candidate-construction-begin-internal!",
        ):
            self.assertIn(name, source)
        self.assertIn("(define (candidate-construction-begin",
                      (ROOT / "native/g3r_candidate_p1_inside.esk").read_text())
        self.assertLess(source.index("(vector-set! g3r-candidate-ledgers 0 next)"),
                        source.index("(define (g3r-candidate-model-create-internal!"))
        self.assertIn("(eq? (vector-ref row 7) 'armed)",
                      (ROOT / "native/g3r_candidate_p1_inside.esk").read_text())
        self.assertLess((ROOT / "native/g3r_candidate_p1_inside.esk").read_text().index(
                        "(vector-set! row 7 'consumed)"),
                        (ROOT / "native/g3r_candidate_p1_inside.esk").read_text().index(
                        "(p1-native-candidate-construction-begin native-context)"))
        self.assertLess(source.index("(set! native (m3t-native-owner-create"),
                        source.index("(vector-set! (vector-ref canonical 4) 0 native)"))
        self.assertEqual(set(re.findall(r"\(vector-set! row (\d+)", source)),
                         {"7", "8"})
        self.assertIn("(vector-set! row 7 'armed)\n"
                      "          (if (= g3r-candidate-test-fault 6)\n"
                      "              (g3r-candidate-test-armed-probes p1-cell row))\n"
                      "          (let ((construction\n"
                      "                  (candidate-construction-begin\n"
                      "                    i2-provider-name p1-cell)))", source)
        self.assertIn("(vector-set! owner-cell 0 foreign)", source)
        self.assertIn("(vector-set! owner-cell 0 owned)", source)
        self.assertIn("(g3r-candidate-test-armed-probes p1-cell row)", source)
        self.assertIn("(g3r-candidate-roster-bytes)", source)
        self.assertIn("(candidate-construction-begin 'foreign-provider p1-cell)", source)
        self.assertIn("(candidate-construction-begin i2-provider-name copied)", source)
        p1 = (ROOT / "native/g3r_candidate_p1_inside.esk").read_text()
        self.assertLess(p1.index("(vector-set! row 7 'consumed)"),
                        p1.index("(g3r-candidate-test-consumed-replay"))
        self.assertLess(p1.index("(g3r-candidate-test-consumed-replay"),
                        p1.index("(p1-native-candidate-construction-begin native-context)"))
        self.assertIn("(candidate-construction-begin provider p1-cell)", source)
        self.assertIn("(eq? (vector-ref row 7) 'consumed)", source)
        self.assertIn("(eq? (vector-ref i2-construction-registry-root 1) i2-record)", source)
        self.assertLess(p1.index("(set! begin-attempted? #t)"),
                        p1.index("(p1-native-candidate-construction-begin native-context)"))
        self.assertLess(p1.index("(p1-native-candidate-construction-begin native-context)"),
                        p1.index("(g3r-candidate-test-cut 7)"))
        self.assertLess(p1.index("(g3r-candidate-test-cut 7)"),
                        p1.index("(set! identity (p1-native-result-ptr native-context))"))
        self.assertLess(p1.index("(g3r-candidate-test-cut 8)"),
                        p1.index("(vector-set! canonical 0 identity)"))
        self.assertIn("(p1-native-result-ptr native-context))\n                                 #f)", p1)
        self.assertLess(source.index("(vector-set! enrollment-cell 3 (vector-ref record 4))"),
                        source.index("(i2-construction-seal-internal! record)"))
        self.assertLess(source.index("(g3r-candidate-test-cut 4)"),
                        source.index("(i2-candidate-construction-seal-internal!\n"))


if __name__ == "__main__":
    unittest.main()
