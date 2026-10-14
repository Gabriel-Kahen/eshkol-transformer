"""Source-only and classifier controls; the fe9 compiled gate is separate."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.e3_private.parse_overflow_observation import (
    LOSS, NORM, ONE, PADDED_CE, PPL, ROLE_NAMES, SHAPES, ZERO,
    expected_role, parse,
)
from tests.e3_reference.finite_overflow import GAMMAS, materialize, validate


ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "tests/e3_private"


def transcript(case: str, gradient: str) -> bytes:
    records = []
    for _call in range(1 if case == "finite" else 2):
        ordinal = 0
        records.append(f"E3ROW\t{ordinal}\t3,0\t4,0\t1,0")
        for name, shape in zip(ROLE_NAMES, SHAPES, strict=True):
            records.append("\t".join(("E3ROLE", str(ordinal), name,
                                      ",".join(map(str, shape)),
                                      ",".join(expected_role(name, case)))))
        records += [
            f"E3CE\t{ordinal}\t{LOSS[case]},{PADDED_CE}",
            f"E3BATCH\t{ordinal}\t{LOSS[case]},{ONE},{ZERO},{ZERO}\t1",
            f"E3STATE\t{ordinal}\t{LOSS[case]},{ONE},{ZERO}\t1,1",
        ]
    if case == "finite":
        records.append(f"E3FINAL\t{LOSS[case]},{ONE},{PPL},{ZERO}\t1,1")
    records.append("E3-FINITE-OVERFLOW-PASS gradient=" + gradient)
    return ("\n".join(records) + "\n").encode("ascii")


class OverflowTupleTests(unittest.TestCase):
    def test_exact_private_tuple_and_mixed_tuple_rejection(self) -> None:
        body = r'''
PROJECT_ROOT=$1
die() { printf '%s\n' "$*" >&2; exit 1; }
raw_private_root=$PROJECT_ROOT/tests/e3_private/overflow_driver_root.esk
raw_package_bridge=$PROJECT_ROOT/tests/e3_private/overflow_bridge.c
raw_package_renames=$PROJECT_ROOT/tests/e3_private/overflow_package_private_renames.txt
raw_public_exports=$PROJECT_ROOT/tests/e3_private/overflow_package_public_exports.txt
raw_include_dirs=("$PROJECT_ROOT/internal/p1/lib" "$PROJECT_ROOT/internal/c1/lib"
  "$PROJECT_ROOT/internal/t1/lib" "$PROJECT_ROOT/src"
  "$PROJECT_ROOT/internal/d2/lib" "$PROJECT_ROOT/internal/e3/lib")
if [[ $2 == mixed ]]; then
  raw_package_bridge=$PROJECT_ROOT/tests/e3_private/retention_bridge.c
fi
source "$PROJECT_ROOT/scripts/e3-private-package-policy.sh"
printf '%s\n' "$e3_tuple_kind"
'''
        good = subprocess.run(["bash", "-c", body, "bash", str(ROOT), "exact"],
                              capture_output=True, text=True, check=False)
        self.assertEqual((good.returncode, good.stdout, good.stderr),
                         (0, "retention-overflow\n", ""))
        bad = subprocess.run(["bash", "-c", body, "bash", str(ROOT), "mixed"],
                             capture_output=True, text=True, check=False)
        self.assertEqual(bad.returncode, 1)
        self.assertIn("mixed", bad.stderr)

    def test_generated_source_stage_requires_fresh_sibling(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-overflow-stage-") as base:
            gate = Path(base)
            stage = gate / ".e3-overflow.Abc123"
            stage.mkdir()
            body = ('source "$1/scripts/e3-mode-graph-stage-cleanup.sh"; '
                    'e3_overflow_generated_stage "$2"')
            args = ["bash", "-c", body, "bash", str(ROOT),
                    str(stage / "e3_overflow_package.o")]
            good = subprocess.run(args, capture_output=True, text=True,
                                  check=False)
            self.assertEqual((good.returncode, good.stdout),
                             (0, str(gate / ".e3-overflow-generated") + "\n"))
            (gate / ".e3-overflow-generated").symlink_to(stage)
            bad = subprocess.run(args, capture_output=True, text=True,
                                 check=False)
            self.assertEqual(bad.returncode, 1)
            self.assertEqual(bad.stdout, "")

    def test_fixture_is_deterministic_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-overflow-static-") as base:
            first = Path(base) / "first"
            second = Path(base) / "second"
            a, b = materialize(first), materialize(second)
            self.assertEqual(a, b)
            self.assertEqual(validate(first), a)
            self.assertEqual(a["checkpoints"]["finite.etcp"]["gamma_word"],
                             f"{GAMMAS['finite']:08x}")
            self.assertEqual(a["corpus"]["start_cursor"],
                             b["corpus"]["start_cursor"])
            path = first / "overflow.etcp"
            damaged = bytearray(path.read_bytes())
            damaged[-1] ^= 1
            path.write_bytes(damaged)
            with self.assertRaises((ValueError, AssertionError)):
                validate(first)

    def test_exact_tuple_delta_and_ordinary_driver_unchanged(self) -> None:
        self.assertEqual(hashlib.sha256((PRIVATE / "retention_driver.esk").read_bytes()).hexdigest(),
                         "fd6abaac261740a3bb6c9b90d1f0a1c01310b9688cebff154db81c0b524d35cb")
        for suffix, added, removed in (
            ("source_closure", {"tests/e3_private/overflow_driver_root.esk",
                                "tests/e3_private/overflow_driver.esk"},
             {"tests/e3_private/retention_driver_root.esk"}),
            ("native_source_closure", {"tests/e3_private/overflow_bridge.c"}, set()),
            ("public_exports", {"et_e3_overflow_run_v1"}, set()),
            ("public_strings", {"et_e3_overflow_run_v1"}, set()),
            ("defined_symbols", {"et_e3_overflow_run_v1"}, set()),
            ("undefined_symbols", {"fflush"}, set()),
        ):
            with self.subTest(suffix=suffix):
                baseline = (PRIVATE / f"retention_package_{suffix}.txt").read_text().splitlines()
                current = (PRIVATE / f"overflow_package_{suffix}.txt").read_text().splitlines()
                self.assertEqual(set(current) - set(baseline), added)
                self.assertEqual(set(baseline) - set(current), removed)
                self.assertEqual(len(current), len(set(current)))
        self.assertEqual((PRIVATE / "overflow_package_archive_members.txt").read_text(),
                         "e3_overflow_package.o\n")
        self.assertEqual((PRIVATE / "overflow_package_native_objects.txt").read_bytes(),
                         (PRIVATE / "retention_package_native_objects.txt").read_bytes())

    def test_raw_counter_observer_is_overflow_only_and_registry_first(self) -> None:
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        guarded = frame.split("#ifdef ET_E3_OVERFLOW_TESTING\n", 1)[1].split("#endif", 1)[0]
        self.assertIn("e3_find(pointer)", guarded)
        self.assertNotIn("e3_admit(pointer)", guarded)
        self.assertIn("frame->published_counts[selector]", guarded)
        self.assertIn("frame->lifecycle != 1", guarded)
        self.assertIn("frame->phase != E3_IDLE", guarded)
        self.assertIn("selector < 0 || selector > 1", guarded)
        retention_flags = builder.split(
            '"${e3_tuple_kind}" == retention-first-eval-mutant ) ]]; then\n'
            '    package_native_cflags+=(-DET_E3_RETENTION_TESTING)', 1)[1]
        retention_flags = retention_flags.split("fi", 1)[0]
        self.assertNotIn("ET_E3_OVERFLOW_TESTING", retention_flags)
        overflow_flags = builder.split(
            '"${e3_tuple_kind}" == retention-overflow ]]; then\n'
            '    package_native_cflags+=(-DET_E3_RETENTION_TESTING', 1)[1]
        overflow_flags = overflow_flags.split("fi", 1)[0]
        self.assertIn("-DET_E3_OVERFLOW_TESTING", overflow_flags)
        required_overflow = builder.split(
            'e3_required_private+=(et_e3_test_first_eval_corrupt_v1)', 1)[1]
        required_overflow = required_overflow.split(
            'if [[ "${e3_tuple_kind}" == retention-overflow ]]; then', 1)[1]
        required_overflow = required_overflow.split("fi", 1)[0]
        self.assertIn("et_e3_test_published_counter_raw_v1", required_overflow)

    def test_full_trace_and_fault_classification(self) -> None:
        for case, calls in (("finite", 1), ("overflow", 2)):
            for gradient in ("absent", "present"):
                with self.subTest(case=case, gradient=gradient):
                    good = transcript(case, gradient)
                    self.assertEqual(parse(good, case, gradient), calls)
                    for bad in (
                        good[:-1],
                        good + b"E3FINAL\t00000000\t0,0\n",
                        good.replace(b"3,0\t4,0", b"3,0\t3,0", 1),
                        good.replace(b"1,2,256", b"1,2,255", 1),
                        good.replace(NORM[case].encode(), b"7f800000", 1),
                        good.replace(b"E3ROW\t0", b"E3ROW\t1", 1),
                    ):
                        with self.assertRaises(ValueError):
                            parse(bad, case, gradient)
                    if case == "overflow":
                        second = good.find(b"E3ROW\t0", good.find(b"E3ROW\t0") + 1)
                        self.assertGreater(second, 0)
                        old_retry_ordinal = good[:second] + \
                            good[second:].replace(b"E3ROW\t0", b"E3ROW\t1", 1)
                        with self.assertRaisesRegex(ValueError, "ordinal"):
                            parse(old_retry_ordinal, case, gradient)


if __name__ == "__main__":
    unittest.main()
