"""Exact-source and selector checks for the bounded N4/R1/R2/fenv union."""

import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "tests/e3_private"
POLICY = ROOT / "scripts/e3-private-package-policy.sh"


def tuple_inputs(kind):
    if kind == "fault":
        stem = "fault"
    elif kind == "minsubnormal":
        stem = "minsubnormal"
    else:
        stem = "fenv"
    prefix = PRIVATE / f"{stem}_package"
    return [PRIVATE / f"{stem}_driver_root.esk",
            PRIVATE / f"{stem}_bridge.c",
            Path(f"{prefix}_private_renames.txt"),
            Path(f"{prefix}_public_exports.txt")]


def select(paths):
    script = r'''
PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$3
raw_package_renames=$4
raw_public_exports=$5
raw_include_dirs=("$1/internal/p1/lib" "$1/internal/c1/lib" "$1/internal/t1/lib" "$1/src" "$1/internal/d2/lib" "$1/internal/e3/lib")
die() { printf '%s\n' "$*" >&2; exit 77; }
source "$6"
printf '%s\n' "$e3_tuple_kind"
'''
    return subprocess.run(
        ["bash", "-c", script, "bash", str(ROOT),
         *(str(path) for path in paths), str(POLICY)],
        capture_output=True, text=True,
    )


class FenvUnionTest(unittest.TestCase):
    def test_accepted_production_blobs_remain_exact(self):
        expected = {
            "native/e3_private_driver_root.esk": "a27a9a3390225b08cdbf2ad0ad80fd49536736f1fc06d5197451cfde839a82cd",
            "tests/e3_private/driver.esk": "5538c33183760a3522933ab35c7eff9684950be92238d8170de95a1014b4ef0b",
            "src/eshkol_transformer/e3_frame.c": "848317acc2d5b08c39143cfa0af0e41acdf9f8ccb3c4b72e10e515d0d2d66864",
        }
        for relative, digest in expected.items():
            with self.subTest(relative=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)

    def test_three_complete_routes_and_cross_tuple_negatives(self):
        for kind in ("fault", "minsubnormal", "fenv"):
            with self.subTest(kind=kind):
                result = select(tuple_inputs(kind))
                self.assertEqual((result.returncode, result.stdout),
                                 (0, kind + "\n"), result.stderr)
        fenv = tuple_inputs("fenv")
        for other in ("fault", "minsubnormal"):
            for index in range(4):
                mixed = fenv.copy()
                mixed[index] = tuple_inputs(other)[index]
                with self.subTest(other=other, index=index):
                    self.assertEqual(select(mixed).returncode, 77)

    def test_generated_stages_are_disjoint_and_fault_variant_is_preserved(self):
        helper = ROOT / "scripts/e3-generated-source-stage.sh"
        with tempfile.TemporaryDirectory() as temporary:
            gate = Path(temporary)
            for kind in ("minsubnormal", "fenv"):
                stage = gate / f".e3-{kind}.ABC123"
                stage.mkdir()
                own = stage / f"e3_{kind}_package.o"
                foreign = stage / f"e3_{'fenv' if kind == 'minsubnormal' else 'minsubnormal'}_package.o"
                function = f"e3_{kind}_generated_stage"
                command = 'source "$1"; "$2" "$3"'
                def run(path):
                    return subprocess.run(["bash", "-c", command, "bash",
                                           str(helper), function, str(path)],
                                          capture_output=True, text=True)
                self.assertEqual(run(own).stdout,
                                 str(gate / f".e3-{kind}-generated") + "\n")
                self.assertNotEqual(run(foreign).returncode, 0)
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn('if [[ "${e3_tuple_kind}" == fault ]]; then\n'
                      '      e3_p1_variant_flags+=(--fault-enter-denial)', builder)
        self.assertEqual(builder.count('"${e3_p1_variant_flags[@]}"'), 2)
        self.assertIn('"${e3_tuple_kind}" == minsubnormal', builder)
        self.assertIn('"${e3_tuple_kind}" == fenv', builder)


if __name__ == "__main__":
    unittest.main()
