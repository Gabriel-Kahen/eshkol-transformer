"""Host checks for the exact, noninstalled real-zero-model E3 tuple."""

from __future__ import annotations

import subprocess
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


if __name__ == "__main__":
    unittest.main()
