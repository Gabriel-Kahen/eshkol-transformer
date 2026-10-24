"""Host checks for the exact, noninstalled real-zero-model E3 tuple."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
import hashlib
import importlib.util
from copy import deepcopy
from math import log
from pathlib import Path

from tests.e3_metrics.reference_probe import accumulate, divide, finalize, reduce_bool
from tests.e3_reference.transcript import (
    TranscriptError, f32_bits, validate_observed_case,
)
from tests.e3_private.parse_zero_control_observation import ROLE_NAMES, parse
from tests.e3_private.test_retention_first_baseline import form


ROOT = Path(__file__).resolve().parents[2]
ZERO = "00000000"


def complete_observation() -> bytes:
    lines = []
    for ordinal, mask in enumerate(("1,1", "1,1", "1,0")):
        lines.append(f"E3ROW\t{ordinal}\t0,0\t0,0\t{mask}")
        lines.extend(
            f"E3ROLE\t{ordinal}\t{name}\t1\t{ZERO}" for name in ROLE_NAMES
        )
        lines.extend((
            f"E3CE\t{ordinal}\t{ZERO},{ZERO}",
            f"E3BATCH\t{ordinal}\t{ZERO},{ZERO},{ZERO},{ZERO}\t{2 if ordinal < 2 else 1}",
            f"E3STATE\t{ordinal}\t{ZERO},{ZERO},{ZERO}\t5,{ordinal + 1}",
        ))
    lines.extend((f"E3FINAL\t{ZERO},{ZERO},{ZERO},{ZERO}\t5,3",
                  "E3-ZERO-CONTROL-RUNTIME-PASS"))
    return ("\n".join(lines) + "\n").encode("ascii")


class ZeroControlContract(unittest.TestCase):
    def test_independent_zero_forward_distinguishes_all_three_target_classes(self) -> None:
        from tests.e3_private.zero_control_reference import create

        with tempfile.TemporaryDirectory(prefix="e3-zero-reference-") as temporary:
            bundle = create(Path(temporary) / "corpus")
        self.assertEqual([case["name"] for case in bundle["cases"]],
                         ["all-correct", "all-wrong", "mixed"])
        self.assertEqual([case["synthetic"]["final"]["metrics_f32"][3]
                          for case in bundle["cases"]],
                         ["3f800000", "00000000", "3f19999a"])
        self.assertTrue(all(word == ZERO for case in bundle["cases"]
                            for batch in case["synthetic"]["batches"]
                            for word in batch["logits_f32"]))

    def test_new_eshkol_driver_forms_are_balanced(self) -> None:
        for name in ("zero_control_driver.esk", "zero_control_negative_driver.esk"):
            with self.subTest(name=name):
                source = (ROOT / "tests/e3_private" / name).read_text()
                cursor = visited = 0
                while "(define " in source[cursor:]:
                    cursor = source.index("(define ", cursor)
                    cursor += len(form(source[cursor:], "(define "))
                    visited += 1
                self.assertEqual(visited, source.count("(define "))

    def test_tuple_manifests_extend_the_current_union(self) -> None:
        base = ROOT / "native/e3_private_package"
        zero = ROOT / "tests/e3_private/zero_control_package"
        additions = {
            "defined_symbols": {"et_e3_zero_control_run_v1",
                                "et_e3_zero_negative_run_v1"},
            "public_exports": {"et_e3_zero_control_run_v1",
                               "et_e3_zero_negative_run_v1"},
            "public_strings": {"et_e3_zero_control_run_v1",
                               "et_e3_zero_negative_run_v1"},
            "native_source_closure": {"tests/e3_private/zero_control_bridge.c"},
            "source_closure": {
                "@BUILD@/source/transformer/module.esk",
                "tests/e3_private/zero_control_driver_root.esk",
                "tests/e3_private/zero_control_driver.esk",
                "tests/e3_private/zero_control_negative_driver.esk",
            },
            "undefined_symbols": {"fflush"},
        }
        for suffix, added in additions.items():
            with self.subTest(suffix=suffix):
                previous = (base.with_name(base.name + "_" + suffix + ".txt")
                            .read_text().splitlines())
                current = (zero.with_name(zero.name + "_" + suffix + ".txt")
                           .read_text().splitlines())
                self.assertEqual(set(current) - set(previous), added)
                self.assertEqual(
                    set(previous) - set(current),
                    {"internal/p1/lib/transformer/module.esk"}
                    if suffix == "source_closure" else set(),
                )
                self.assertEqual(len(current), len(set(current)))

    def test_observer_requires_complete_ordered_actual_roles(self) -> None:
        raw = complete_observation()
        observed, roles = parse(raw)
        self.assertEqual(len(roles.splitlines()), 63)
        self.assertEqual(observed["batches"][2]["mask"], [True, False])
        for mutated in (
            raw.replace(b"E3ROLE\t0\tET", b"E3ROLE\t0\tZ", 1),
            raw.replace(b"E3ROLE\t1\tET", b"E3ROLE\t2\tET", 1),
            raw.replace(b"E3CE\t0", b"E3CE\t3", 1),
            raw.replace(b"E3BATCH\t2\t00000000,00000000,00000000,00000000\t1",
                        b"E3BATCH\t2\t00000000,00000000,00000000,00000000\t2", 1),
            raw.replace(b"E3FINAL\t", b"E3OTHER\t", 1),
            raw.replace(b"E3ROLE\t0\tET\t1\t00000000",
                        b"E3ROLE\t0\tET\t1\t3f800000", 1),
            raw[:-1],
        ):
            with self.subTest(mutated=mutated[-65:]):
                with self.assertRaises(ValueError):
                    parse(mutated)

    def test_numeric_transcript_mutants_reject_offline(self) -> None:
        ce = f32_bits(log(256.0))
        numerator, weight = reduce_bool((int(ce, 16), int(ce, 16)),
                                        (True, True))
        state = accumulate((0, 0, 0, 0, 0),
                           (numerator, weight, int("40000000", 16), 2))
        original = {
            "kind": "observed-f32", "profile": "e3-private-n1-t2-v256-d4-v1",
            "batches": [{
                "ordinal": 0, "inputs": [0, 0], "targets": [0, 0],
                "mask": [True, True], "logits_f32": [ZERO] * 512,
                "ce_f32": [ce, ce],
                "reduction_f32": [f"{numerator:08x}", f"{weight:08x}",
                                  f"{divide(numerator, weight):08x}"],
                "correct_f32": "40000000",
                "state_f32": [f"{word:08x}" for word in state[:3]],
                "counters": list(state[3:]),
            }],
            "final": {"metrics_f32": [f"{word:08x}" for word in finalize(state)],
                      "counters": list(state[3:])},
        }
        expected = ({"inputs": [0, 0], "targets": [0, 0],
                     "mask": [True, True]},)
        validate_observed_case(original, expected)
        for field, index, replacement in (
            ("logits_f32", 0, "3f800000"),
            ("ce_f32", 0, ZERO),
            ("reduction_f32", 0, ZERO),
            ("state_f32", 0, ZERO),
            ("correct_f32", None, ZERO),
        ):
            with self.subTest(field=field):
                damaged = deepcopy(original)
                if index is None:
                    damaged["batches"][0][field] = replacement
                else:
                    damaged["batches"][0][field][index] = replacement
                with self.assertRaises(TranscriptError):
                    validate_observed_case(damaged, expected)
        damaged = deepcopy(original)
        damaged["final"]["metrics_f32"][0] = ZERO
        with self.assertRaises(TranscriptError):
            validate_observed_case(damaged, expected)

    def test_stable_generated_source_stage_is_owned_and_fresh(self) -> None:
        helper = ROOT / "scripts/e3-zero-generated-source-stage.sh"
        with tempfile.TemporaryDirectory() as temporary:
            gate = Path(temporary)
            stage = gate / ".e3-zero-control.ABC123"
            stage.mkdir()
            object_path = stage / "e3_zero_control_package.o"

            def checked(path: Path) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["bash", "-c", 'source "$1"; e3_zero_generated_stage "$2"',
                     "stage-check", str(helper), str(path)],
                    text=True, capture_output=True, check=False,
                )

            self.assertEqual(checked(object_path).stdout.strip(),
                             str(gate / ".e3-zero-control-generated"))
            self.assertNotEqual(checked(stage / "wrong.o").returncode, 0)
            (gate / ".e3-zero-control-generated").symlink_to(stage)
            self.assertNotEqual(checked(object_path).returncode, 0)

    def test_mixed_package_tuple_rejected(self) -> None:
        policy = ROOT / "scripts/e3-private-package-policy.sh"
        base = ROOT / "native/e3_private_package"
        zero = ROOT / "tests/e3_private/zero_control_package"
        script = '''
          die() { printf '%s\\n' "$*" >&2; exit 41; }
          PROJECT_ROOT=$1
          raw_private_root=$2
          raw_package_bridge=$3
          raw_package_renames=$4
          raw_public_exports=$5
          source "$6"
        '''
        result = subprocess.run(
            ["bash", "-c", script, "policy-check", str(ROOT),
             str(base.with_name("e3_private_driver_root.esk")),
             str(ROOT / "tests/e3_private/zero_control_bridge.c"),
             str(zero) + "_private_renames.txt",
             str(zero) + "_public_exports.txt", str(policy)],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 41)
        self.assertIn("mixed", result.stderr)

    def test_14_unique_model_paths_and_postload_encoder(self) -> None:
        driver = (ROOT / "tests/e3_private/zero_control_driver.esk").read_text()
        self.assertEqual(driver.count('("blocks" "0" "attention"'), 8)
        self.assertIn('(eq? (parameter-tree-handle tree \'("head" "weight"))', driver)
        self.assertIn('(i2-checkpoint-decode request)', driver)
        self.assertIn('(i2-commit-load! (vector-ref prepared 0))', driver)
        self.assertIn('(i2-checkpoint-encode', driver)
        self.assertLess(driver.index('(e3-zero-load! handles sources)'),
                        driver.index('(e3-frame-bind! model destinations)'))
        self.assertNotIn('e3-test-run scenario', driver)

    def test_prepared_plan_has_no_observation_or_recovery_tail(self) -> None:
        driver = (ROOT / "tests/e3_private/zero_control_driver.esk").read_text()

        def covered(source: str) -> bool:
            load = source.split('(define (e3-zero-load! handles sources)', 1)[1].split(
                '(define (e3-zero-check-bytes! handles)', 1)[0]
            commit = '(i2-commit-load! (vector-ref prepared 0))'
            cleanup = '(e3-zero-release-sources! sources)'
            if (load.count('(i2-prepare-load') != 1 or load.count(commit) != 1
                    or load.count(cleanup) != 2
                    or '(guard (caught (#t ' + cleanup not in load):
                return False
            prepared = load.index('(i2-prepare-load')
            committed = load.index(commit)
            return (prepared < committed < load.rindex(cleanup)
                    and load.index(cleanup) < prepared
                    and not any(fragment in load[prepared:committed]
                                for fragment in ('(e3-zero-invariant',
                                                 '(e3-zero-gradient', '(guard ')))

        self.assertTrue(covered(driver))
        self.assertFalse(covered(driver.replace(
            '(i2-commit-load! (vector-ref prepared 0))',
            '(i2-commit-load! prepared)', 1)))
        self.assertFalse(covered(driver.replace(
            '    (e3-zero-release-sources! sources)\n    (vector-ref prepared 0)))',
            '    (vector-ref prepared 0)))', 1)))
        self.assertNotIn('model-release', driver)

    def test_negative_matrix_uses_genuine_foreign_and_tied_head(self) -> None:
        driver = (ROOT / "tests/e3_private/zero_control_negative_driver.esk").read_text()
        runner = (ROOT / "scripts/test-e3-zero-control.sh").read_text()
        self.assertIn('(module-parameters foreign)', driver)
        self.assertIn('(e3-zero-require-tree-handle tree 0 foreign-handle)', driver)
        self.assertNotIn("(vector 'not-a-p1-handle)", driver)
        self.assertIn("'(\"token_embedding\" \"weight\")", driver)
        self.assertIn('(vector-ref sources 10)', driver)
        for name in ("foreign-tree", "duplicate-tied-head", "wrong-rank",
                     "wrong-f32-geometry", "decode-request-length",
                     "decode-payload", "decode-shape", "decode-rank-limit",
                     "decode-dtype", "decode-device", "decode-layout",
                     "decode-tail", "decode-operation"):
            with self.subTest(name=name):
                self.assertIn(f'((string=? name "{name}")', driver)
                self.assertIn(name, runner)

    def test_stale_fixture_is_aborted_before_zero_input_admission(self) -> None:
        negative = (ROOT / "tests/e3_private/zero_control_negative_driver.esk").read_text()
        positive = (ROOT / "tests/e3_private/zero_control_driver.esk").read_text()
        runner = (ROOT / "scripts/test-e3-zero-control.sh").read_text()
        fixture = negative.split('(define (e3-zero-negative-stale! tree)', 1)[1].split(
            '(define (e3-zero-negative-fire', 1)[0]
        self.assertLess(fixture.index('(i2-module-register-parameter-internal!'),
                        fixture.index('(i2-construction-abort-internal! construction))\n'
                                      '              (set! aborted #t)'))
        self.assertLess(fixture.index('(set! aborted #t)'),
                        fixture.index('(e3-zero-require-tree-handle tree 0 stale)'))
        self.assertIn('(transformer-error-category stale-error)', fixture)
        self.assertIn("'invalid-state", fixture)
        self.assertIn('(e3-zero-stale-owner-abort owner)', fixture)
        admission = positive.split('(define (e3-zero-require-tree-handle', 1)[1].split(
            '(define (e3-zero-stage!', 1)[0]
        self.assertLess(admission.index('(parameter-handle-shape handle)'),
                        admission.index('(parameter-tree-handle tree'))
        self.assertNotIn('(parameter-handle-tensor-internal', admission)
        fire = negative.split('(define (e3-zero-negative-fire', 1)[1].split(
            '(define (e3-zero-negative-run', 1)[0]
        self.assertLess(fire.index('(e3-zero-negative-stale! tree)'),
                        fire.index('(parameter-handle-tensor-internal'))
        self.assertIn('(if (string=? name "stale-handle")', fire)
        self.assertIn('stale-handle', runner)

    def test_zero_p1_overlay_is_one_pinned_insertion(self) -> None:
        path = ROOT / "scripts/generate-e3-zero-p1-topology-overlay.py"
        spec = importlib.util.spec_from_file_location("zero_p1_overlay", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = module.CANONICAL.read_bytes()
        emitted = module.compose(source)
        module.verify_manifest()
        self.assertEqual(hashlib.sha256(source).hexdigest(), module.SOURCE_SHA256)
        self.assertEqual(hashlib.sha256(emitted).hexdigest(), module.EMITTED_SHA256)
        self.assertEqual(emitted.count(module.INSERTION), 1)
        self.assertEqual(emitted.replace(module.INSERTION, b"", 1), source)
        self.assertIn("Zero-control-only P1 projection",
                      form(emitted.decode("utf-8"), "(define p1-trusted-surface"))
        self.assertIn(b"(lambda (model modules handles carriers selector)",
                      module.INSERTION)
        self.assertNotIn(b"(module-parameters", module.INSERTION)
        for phrase in (b"'path-10", b"'shape-10", b"'tie-10",
                       b"'throw-path-10", b"(vector-set! target field original)",
                       b"(tr3-fixed-recheck!"):
            self.assertIn(phrase, module.INSERTION)

        def rollback_control(body: bytes) -> bool:
            return (
                body.count(b"(vector-set! target field original)") == 2
                and body.count(b"(vector-set! target field replacement)") == 1
                and body.count(b"(raise 'e3-zero-topology-test-marker)") == 1
                and body.count(b"(if (not (= mismatch 4))") == 1
                and body.count(b"(if (not (eq? (vector-ref target field) original))") == 2
                and b"(module-parameters" not in body
                and body.index(b"(guard (caught") <
                    body.index(b"(vector-set! target field replacement)") <
                    body.rindex(b"(vector-set! target field original)")
            )

        self.assertTrue(rollback_control(module.INSERTION))
        for damaged in (
            module.INSERTION.replace(b"(vector-set! target field original)",
                                     b"(begin #t)", 1),
            module.INSERTION.replace(b"(if (not (= mismatch 4))",
                                     b"(if (not (= mismatch 3))", 1),
            module.INSERTION.replace(b"(raise 'e3-zero-topology-test-marker)",
                                     b"(begin #t)", 1),
            module.INSERTION.replace(b"(vector-set! target field replacement)",
                                     b"(module-parameters model)\n"
                                     b"                        (vector-set! target field replacement)", 1),
        ):
            self.assertFalse(rollback_control(damaged))
        with self.assertRaises(ValueError):
            module.compose(source + b"\n")
        original_sha = module.EMITTED_SHA256
        try:
            module.EMITTED_SHA256 = "0" * 64
            with self.assertRaises(ValueError):
                module.compose(source)
        finally:
            module.EMITTED_SHA256 = original_sha
        with tempfile.TemporaryDirectory() as temporary:
            damaged_manifest = Path(temporary) / "manifest.json"
            damaged_manifest.write_text('{"kind":"forged"}\n')
            actual_manifest = module.MANIFEST
            try:
                module.MANIFEST = damaged_manifest
                with self.assertRaises(ValueError):
                    module.verify_manifest()
            finally:
                module.MANIFEST = actual_manifest
            output = Path(temporary) / "source/transformer/module.esk"
            command = ["python3", "-B", str(path), "--output", str(output)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run(command + ["--check"],
                                            capture_output=True).returncode, 0)
            self.assertNotEqual(subprocess.run(
                ["python3", "-B", str(path), "--output",
                 f"{output.parent}/./module.esk", "--check"],
                capture_output=True).returncode, 0)
            output.write_bytes(emitted + b"\n")
            self.assertNotEqual(subprocess.run(command + ["--check"],
                                               capture_output=True).returncode, 0)
            output.unlink()
            output.symlink_to(module.CANONICAL)
            self.assertNotEqual(subprocess.run(command + ["--check"],
                                               capture_output=True).returncode, 0)

    def test_zero_overlay_policy_rejects_other_tuples_and_aliases(self) -> None:
        generator = ROOT / "scripts/generate-e3-zero-p1-topology-overlay.py"
        policy = ROOT / "scripts/e3-private-package-policy.sh"
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "source"
            overlay = stage / "transformer/module.esk"
            subprocess.run(["python3", "-B", str(generator), "--output",
                            str(overlay)], check=True, capture_output=True)
            script = '''
              set -o pipefail
              die() { printf '%s\\n' "$*" >&2; exit 41; }
              PROJECT_ROOT=$1
              raw_private_root=$2
              raw_package_bridge=$3
              raw_package_renames=$4
              raw_public_exports=$5
              raw_include_dirs=("$PROJECT_ROOT/internal/p1/lib"
                "$PROJECT_ROOT/internal/c1/lib" "$PROJECT_ROOT/internal/t1/lib"
                "$PROJECT_ROOT/src" "$PROJECT_ROOT/internal/d2/lib"
                "$PROJECT_ROOT/internal/e3/lib")
              if [[ "${10}" == forged ]]; then
                raw_include_dirs+=("$8")
              fi
              source "$6"
              e3_tuple_kind=$7
              e3_generated_source_root=$8
              printf '%s\\n' "$9" | e3_normalize_source_dependencies
            '''

            def check(kind: str, dependency: Path | str,
                      *, mixed: bool = False,
                      forged_include: bool = False) -> subprocess.CompletedProcess[str]:
                zero = ROOT / "tests/e3_private/zero_control_package"
                inputs = [ROOT / "tests/e3_private/zero_control_driver_root.esk",
                          ROOT / "tests/e3_private/zero_control_bridge.c",
                          Path(str(zero) + "_private_renames.txt"),
                          Path(str(zero) + "_public_exports.txt")]
                if mixed:
                    inputs[1] = ROOT / "native/e3_private_bridge.c"
                return subprocess.run(
                    ["bash", "-c", script, "policy-check", str(ROOT),
                     *(str(item) for item in inputs), str(policy), kind,
                     str(stage), str(dependency),
                     "forged" if forged_include else "exact"],
                    capture_output=True, text=True, check=False,
                )

            self.assertEqual(check("zero", overlay).stdout.strip(),
                             "@BUILD@/source/transformer/module.esk")
            for kind in ("base", "diagnostic", "public", "retention",
                         "retention-first-eval-mutant", "retention-overflow",
                         "mode-graph", "numerical", "uniform", "partial"):
                with self.subTest(kind=kind):
                    self.assertNotEqual(check(kind, overlay).returncode, 0)
            self.assertNotEqual(check("zero", f"{overlay.parent}/./module.esk").returncode, 0)
            self.assertNotEqual(check("zero", overlay, mixed=True).returncode, 0)
            self.assertNotEqual(check("zero", overlay,
                                      forged_include=True).returncode, 0)
            authentic = Path(temporary) / "authentic-module.esk"
            authentic.write_bytes(overlay.read_bytes())
            overlay.unlink()
            overlay.symlink_to(authentic)
            self.assertNotEqual(check("zero", overlay).returncode, 0)

    def test_topology_probe_restores_before_observing_tree(self) -> None:
        negative = (ROOT / "tests/e3_private/zero_control_negative_driver.esk").read_text()
        runner = (ROOT / "scripts/test-e3-zero-control.sh").read_text()
        build = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        probe = negative.split('(define (e3-zero-negative-topology! model handles name)', 1)[1].split(
            '(define (e3-zero-negative-fire', 1)[0]
        self.assertIn('(e3-zero-negative-p1-shell-census)', negative)
        self.assertIn('(e3-zero-negative-topology-shells? model modules)', probe)
        self.assertIn('(e3-zero-negative-topology-pairs? model retained carriers)', probe)
        self.assertLess(probe.index('(vector-ref p1-trusted-surface 69)'),
                        probe.index('(vector-ref p1-trusted-surface 73)'))
        self.assertLess(probe.index('(vector-ref p1-trusted-surface 73)'),
                        probe.index('(e3-zero-check-tree model)'))
        for name in ("topology-path-10", "topology-shape-10",
                     "topology-tie-10", "topology-throw-path-10"):
            self.assertIn(name, negative)
            self.assertIn(name, runner)
        self.assertIn('scripts/generate-e3-zero-p1-topology-overlay.py', build)
        self.assertIn('if [[ "${e3_tuple_kind}" == zero ]]; then\n'
                      '    include_args=(-I "${e3_generated_source_root}"', build)

    def test_prebind_source_cleanup_precedes_seed_and_destination_allocation(self) -> None:
        driver = (ROOT / "tests/e3_private/zero_control_negative_driver.esk").read_text()

        def covered(source: str) -> bool:
            stage = source.index('(e3-zero-stage! tree handles sources)')
            guard = source.index('(guard (caught', stage)
            seed = source.index('(if gradient-present?', stage)
            destination = source.index('(e3-test-destinations-create)', stage)
            cleanup = source.index('(e3-zero-release-sources! sources)', guard)
            depth = 0
            end = guard
            for end in range(guard, len(source)):
                if source[end] == '(':
                    depth += 1
                elif source[end] == ')':
                    depth -= 1
                    if depth == 0:
                        break
            return stage < guard < cleanup < seed < destination < end

        self.assertTrue(covered(driver))
        damaged = driver.replace(
            '(e3-zero-release-sources! sources)', '(begin #t)', 1
        )
        self.assertFalse(covered(damaged))

    def test_negative_byte_image_uses_its_defined_comparator(self) -> None:
        driver = (ROOT / "tests/e3_private/zero_control_negative_driver.esk").read_text()
        self.assertIn("(define (e3-zero-negative-bytes=? left right)", driver)
        self.assertEqual(driver.count("(e3-zero-negative-bytes=?"), 3)
        self.assertNotIn("(e3-zero-negative-bytes?", driver)


if __name__ == "__main__":
    unittest.main()
