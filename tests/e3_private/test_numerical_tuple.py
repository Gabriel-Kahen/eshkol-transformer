"""Static exact-tuple and compiled-mutant boundary checks."""

from pathlib import Path
import subprocess
import unittest

from tests.e3_private.parse_numerical_observation import ROLE_NAMES, parse


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "native/e3_private_package"
NUMERICAL = ROOT / "tests/e3_private/numerical_package"


def lines(prefix: Path, suffix: str) -> list[str]:
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


class NumericalTupleTest(unittest.TestCase):
    def test_exact_noninstalled_successor_inventories(self) -> None:
        for suffix in (
            "native_objects", "defined_symbols", "public_exports",
            "public_strings", "private_renames",
        ):
            self.assertEqual(lines(NUMERICAL, suffix), lines(BASE, suffix), suffix)
        self.assertEqual(lines(NUMERICAL, "undefined_symbols"),
                         sorted(lines(BASE, "undefined_symbols") + ["fflush"]))
        self.assertEqual(
            lines(NUMERICAL, "source_closure"),
            ["tests/e3_private/numerical_driver_root.esk"]
            + lines(BASE, "source_closure"),
        )
        native = lines(BASE, "native_source_closure")
        native.insert(native.index("native/e3_private_bridge.c"),
                      "tests/e3_private/numerical_bridge.c")
        self.assertEqual(lines(NUMERICAL, "native_source_closure"), native)
        self.assertEqual(lines(NUMERICAL, "archive_members"),
                         ["e3_numerical_package.o"])
        self.assertNotIn("numerical_driver_root.esk",
                         (ROOT / "native/e3_private_driver_root.esk").read_text())

    def test_policy_rejects_mixed_tuple_before_compilation(self) -> None:
        policy = ROOT / "scripts/e3-private-package-policy.sh"
        root = ROOT / "tests/e3_private/numerical_driver_root.esk"
        bridge = ROOT / "tests/e3_private/numerical_bridge.c"
        renames = Path(f"{NUMERICAL}_private_renames.txt")
        exports = Path(f"{NUMERICAL}_public_exports.txt")
        script = '''
PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$3
raw_package_renames=$4
raw_public_exports=$5
raw_include_dirs=("${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib")
die() { printf '%s\\n' "$*" >&2; exit 77; }
source "$6"
printf '%s\\n' "$e3_tuple_kind"
'''
        exact = subprocess.run(
            ["bash", "-c", script, "bash", str(ROOT), str(root), str(bridge),
             str(renames), str(exports), str(policy)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual((exact.returncode, exact.stdout), (0, "numerical\n"),
                         exact.stderr)
        mixed = subprocess.run(
            ["bash", "-c", script, "bash", str(ROOT), str(root),
             str(ROOT / "native/e3_private_bridge.c"), str(renames),
             str(exports), str(policy)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(mixed.returncode, 77)
        self.assertIn("mixed", mixed.stderr)

    def test_observer_and_mutant_guards_are_exact(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        metrics = (ROOT / "native/e3_evaluation_metrics_provider.c").read_text()
        self.assertIn('"${e3_tuple_kind}" == numerical', builder)
        self.assertIn("-DET_E3_NUMERICAL_TESTING", builder)
        self.assertIn("-DET_E3_NUMERICAL_WRONG_TARGET", builder)
        self.assertIn("-DET_E3_NUMERICAL_MEAN_OF_MEANS", builder)
        self.assertIn("#ifdef ET_E3_NUMERICAL_TESTING", frame)
        self.assertIn("#ifdef ET_E3_NUMERICAL_WRONG_TARGET", frame)
        self.assertIn("#ifdef ET_E3_NUMERICAL_MEAN_OF_MEANS", metrics)
        self.assertNotIn("E3ROW", (ROOT / "tests/e3_private/driver.esk").read_text())

    def test_incomplete_compiled_stdout_cannot_become_observation(self) -> None:
        with self.assertRaisesRegex(ValueError, "did not complete"):
            parse(b"E3ROW\t0\t3,197\t197,0\t1,1\n")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            parse(b"reference JSON instead of native frame\n"
                  b"E3-NUMERICAL-RUNTIME-PASS\n")

    def test_parser_uses_tagged_frame_words_without_oracle_substitution(self) -> None:
        lines = []
        for ordinal in range(3):
            lines.append(f"E3ROW\t{ordinal}\t3,197\t197,0\t1,1")
            for name in ROLE_NAMES:
                shape = "1,2,256" if name == "Z" else "1"
                words = ",".join(["3f800000"] * (512 if name == "Z" else 1))
                lines.append(f"E3ROLE\t{ordinal}\t{name}\t{shape}\t{words}")
            lines.extend((
                f"E3CE\t{ordinal}\t3f800000,40000000",
                f"E3BATCH\t{ordinal}\t40400000,40000000,3fc00000,00000000\t2",
                f"E3STATE\t{ordinal}\t40400000,40000000,00000000\t{2 * (ordinal + 1)},{ordinal + 1}",
            ))
        lines.extend(("E3FINAL\t3f800000,40000000,40400000,00000000\t6,3",
                      "E3-NUMERICAL-RUNTIME-PASS"))
        payload, roles = parse(("\n".join(lines) + "\n").encode("ascii"))
        self.assertEqual(payload["batches"][0]["inputs"], [3, 197])
        self.assertEqual(payload["batches"][2]["targets"], [197, 0])
        self.assertEqual(payload["batches"][1]["ce_f32"],
                         ["3f800000", "40000000"])
        self.assertEqual(payload["final"]["metrics_f32"][0], "3f800000")
        self.assertEqual(len(roles.splitlines()), 63)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse(("\n".join(lines[:-1] + [lines[0], lines[-1]]) + "\n").encode("ascii"))


if __name__ == "__main__":
    unittest.main()
