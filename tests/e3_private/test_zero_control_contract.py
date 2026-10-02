"""Host checks for the exact, noninstalled real-zero-model E3 tuple."""

from __future__ import annotations

import subprocess
import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path

from tests.e3_private.parse_zero_control_observation import ROLE_NAMES, parse


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
    def test_p1_overlay_is_exact_single_slot_and_fails_closed(self) -> None:
        script = ROOT / "scripts/generate-e3-zero-p1-overlay.py"
        spec = importlib.util.spec_from_file_location("zero_p1_overlay", script)
        assert spec and spec.loader
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        canonical = generator.SOURCE.read_bytes()
        overlay = generator.rendered()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(),
                         generator.SOURCE_SHA256)
        self.assertEqual(overlay.count(b"(e3-zero-p1-invariant-fail)"), 3)
        self.assertEqual(
            overlay.replace(generator.SLOT.encode(), generator.ANCHOR.encode())
                   .replace((generator.IMPORT_ANCHOR + generator.IMPORT).encode(),
                            generator.IMPORT_ANCHOR.encode()), canonical)
        self.assertIn(generator.IMPORT.encode(), overlay)
        self.assertIn(b"(vector-set! target field original)", overlay)
        self.assertIn(b"(tr3-fixed-recheck! model modules handles carriers)",
                      overlay)
        self.assertIn(b"(wrong-path '(\"head\" \"wrong\"))", overlay)
        self.assertIn(b"(wrong-shape '(255 4))", overlay)
        self.assertIn(b"(eq? (vector-ref binding 2) head)", overlay)
        self.assertIn(b"(if (not (= status 4))", overlay)
        self.assertNotIn(b"(fail ", generator.SLOT.encode())
        self.assertNotIn(b"(module-parameters model)", generator.SLOT.encode())
        original_slot = generator.SLOT
        try:
            generator.SLOT = original_slot.replace(
                "(e3-zero-p1-invariant-fail)",
                "(fail 'internal 'e3-zero-projection-test \"defect\" '())", 1)
            with self.assertRaisesRegex(ValueError, "defect path must abort"):
                generator.rendered()
        finally:
            generator.SLOT = original_slot
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            command = ["python3", str(script), "--output-dir", str(directory)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode,
                             0)
            self.assertEqual(subprocess.run(command + ["--check"],
                                            capture_output=True).returncode, 0)
            self.assertNotEqual(subprocess.run(command,
                                               capture_output=True).returncode, 0)
            generated = directory / "p1/transformer/module.esk"
            generated.write_bytes(overlay + b"; corrupted\n")
            self.assertNotEqual(subprocess.run(command + ["--check"],
                                               capture_output=True).returncode, 0)
            generator.SOURCE = generated
            with self.assertRaisesRegex(ValueError, "canonical P1 module bytes"):
                generator.rendered()

    def test_projection_source_closure_and_real_negative_routes(self) -> None:
        closure = (ROOT / "tests/e3_private/zero_control_package_source_closure.txt"
                   ).read_text().splitlines()
        self.assertEqual(closure.count("@BUILD@/p1/transformer/module.esk"), 1)
        self.assertNotIn("internal/p1/lib/transformer/module.esk", closure)
        self.assertEqual(
            closure.count("native/tr3_p1_fixed_set_extension.esk"), 1)
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        policy = (ROOT / "scripts/e3-private-package-policy.sh").read_text()
        negative = (ROOT / "tests/e3_private/zero_control_negative_driver.esk"
                    ).read_text()
        runner = (ROOT / "scripts/test-e3-zero-control.sh").read_text()
        self.assertIn('generate-e3-zero-p1-overlay.py', builder)
        self.assertIn('include_args+=(-I "${e3_zero_generated_dir}/p1")', builder)
        self.assertIn("@BUILD@/p1/transformer/module.esk", policy)
        for case in ("foreign-p1-handle", "foreign-p1-module",
                     "foreign-p1-carrier", "stale-p1-handle", "p1-path-10",
                     "p1-shape-10", "p1-tie-10", "p1-census-borrow",
                     "p1-census-plan"):
            self.assertIn(case, negative)
            self.assertIn(case, runner)
        self.assertIn("i2-construction-abort-internal! construction", negative)
        self.assertIn("(e3-zero-staged-owner-abort owner)", negative)
        self.assertIn("(e3-zero-control-model-idle", negative)
        native = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        self.assertIn("idle_fields[] = {0, 1, 2, 3, 4, 6, 7, 8, 10}", native)
        self.assertIn("et_e3_zero_control_census_v1(pointer, idle_fields[index])", native)
        self.assertIn("et_f32_tensor_test_plan_pins_v1(value, &pins)", native)
        self.assertIn("et_f32_tensor_borrow_begin_v1((et_f32_tensor *)destination", native)
        self.assertIn("et_f32_tensor_copy_plan_prepare_v1(1u, &assignment", native)
        self.assertIn("(e3-zero-census-probe-begin owner 0 pointer kind)", negative)
        self.assertIn("(e3-zero-census-probe-end owner 0 pointer kind)", negative)
        self.assertIn("(e3-zero-negative-bytes=? source-before", negative)
        self.assertIn("(e3-zero-control-borrow-begin owner 0 pointer)", negative)
        self.assertIn("(e3-zero-control-pin-begin owner 0 pointer)", negative)
        self.assertIn("(not (= (e3-zero-control-model-idle owner) 0))", negative)

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
            raw[:-1],
        ):
            with self.subTest(mutated=mutated[-65:]):
                with self.assertRaises(ValueError):
                    parse(mutated)

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
        load = driver.split('(define (e3-zero-load! handles sources)', 1)[1].split(
            '(define (e3-zero-check-bytes! handles)', 1)[0]
        self.assertIn('(guard (caught (#t (e3-zero-release-sources! sources)', load)
        prepared = load.index('(i2-prepare-load')
        committed = load.index('(i2-commit-load! (vector-ref prepared 0))')
        tail = load[prepared:committed]
        self.assertNotIn('(e3-zero-invariant', tail)
        self.assertNotIn('(e3-zero-gradient', tail)
        self.assertNotIn('(guard ', tail)
        self.assertGreater(load.index('(e3-zero-release-sources! sources)', committed),
                           committed)
        self.assertNotIn('model-release', driver)

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
        self.assertEqual(driver.count("(e3-zero-negative-bytes=?"), 4)
        self.assertNotIn("(e3-zero-negative-bytes?", driver)


if __name__ == "__main__":
    unittest.main()
