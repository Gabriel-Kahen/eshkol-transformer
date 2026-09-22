from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / "native" / "o2_wave2_package_bridge.c"
EXTENSION = ROOT / "native" / "tr3_o2_step_clear_extension.esk"
BASELINE = "35ccace:native/o2_wave2_package_bridge.c"

PRIVATE_BRIDGE_SYMBOLS = {
    "et_tr3_private_o2_trainer_step_clear_prepare_v1",
    "et_tr3_private_o2_trainer_step_clear_commit_v1",
    "et_tr3_private_o2_trainer_step_clear_abort_v1",
}
CORE_SYMBOLS = {
    "et_o2_trainer_step_clear_prepare_v1",
    "et_o2_trainer_step_clear_commit_v1",
    "et_o2_trainer_step_clear_abort_v1",
}


def run(*args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        args,
        cwd=ROOT,
        input=input_bytes,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def compile_stdin(source: bytes, output: Path, *defines: str) -> None:
    compiler = os.environ.get("CC", "clang")
    run(
        compiler,
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wpedantic",
        "-DET_O2_NATIVE_HELPERS_ONLY",
        *(f"-D{define}" for define in defines),
        "-Iinclude",
        "-Inative",
        "-x",
        "c",
        "-c",
        "-",
        "-o",
        str(output),
        input_bytes=source,
    )


def nm_symbols(obj: Path, flag: str) -> set[str]:
    output = run("nm", flag, "--format=posix", str(obj)).decode()
    return {line.split()[0] for line in output.splitlines() if line.strip()}


class Tr3O2StepClearPackageTests(unittest.TestCase):
    def test_source_adapter_has_only_the_frozen_private_surface(self) -> None:
        source = EXTENSION.read_text()
        for symbol in PRIVATE_BRIDGE_SYMBOLS:
            self.assertEqual(source.count(":real " + symbol), 1)
        self.assertIn("(define (o2-trainer-step-clear-prepare-internal\n", source)
        self.assertIn("(define (o2-trainer-step-clear-commit-internal! plan)", source)
        self.assertIn("(define (o2-trainer-step-clear-abort-internal! plan)", source)
        self.assertNotIn("provide", source)
        self.assertNotIn("export", source)

    def test_predecessor_roots_and_manifests_do_not_name_the_authority(self) -> None:
        paths = [
            ROOT / "native" / f"{package}_wave2_{kind}.txt"
            for package in ("o2", "c2")
            for kind in (
                "defined_symbols",
                "native_source_closure",
                "private_renames",
                "public_exports",
                "public_strings",
                "source_closure",
                "undefined_symbols",
            )
        ]
        paths += [
            ROOT / "native" / "o2_wave2_root.esk",
            ROOT / "native" / "c2_wave2_root.esk",
        ]
        for path in paths:
            text = path.read_text()
            self.assertNotIn("step-clear", text, path)
            self.assertNotIn("step_clear", text, path)

        expected_counts = {
            "o2_wave2_defined_symbols.txt": 53,
            "o2_wave2_public_exports.txt": 47,
            "o2_wave2_public_strings.txt": 53,
            "o2_wave2_private_renames.txt": 6,
            "o2_wave2_undefined_symbols.txt": 150,
            "o2_wave2_source_closure.txt": 15,
            "o2_wave2_native_source_closure.txt": 29,
            "c2_wave2_defined_symbols.txt": 81,
            "c2_wave2_public_exports.txt": 75,
            "c2_wave2_public_strings.txt": 81,
            "c2_wave2_private_renames.txt": 6,
            "c2_wave2_source_closure.txt": 32,
            "c2_wave2_native_source_closure.txt": 53,
        }
        for name, expected in expected_counts.items():
            lines = (ROOT / "native" / name).read_text().splitlines()
            entries = [
                line for line in lines
                if line.strip() and not line.lstrip().startswith("#")
            ]
            self.assertEqual(len(entries), expected, name)

    def test_gate_absent_o2_and_c2_bridge_objects_match_baseline_bytes(self) -> None:
        baseline = run("git", "show", BASELINE)
        current = BRIDGE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            for label, defines in (
                ("o2", ()),
                ("c2", ("ET_C2_O2_RECONSTRUCT_BRIDGE",)),
            ):
                old_obj = temp / f"old-{label}.o"
                new_obj = temp / f"new-{label}.o"
                compile_stdin(baseline, old_obj, *defines)
                compile_stdin(current, new_obj, *defines)
                self.assertEqual(old_obj.read_bytes(), new_obj.read_bytes(), label)
                names = nm_symbols(new_obj, "-g")
                self.assertTrue(names.isdisjoint(PRIVATE_BRIDGE_SYMBOLS | CORE_SYMBOLS))

    def test_gate_adds_exact_bridge_definitions_and_core_references(self) -> None:
        current = BRIDGE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            absent = temp / "absent.o"
            present = temp / "present.o"
            compile_stdin(current, absent)
            compile_stdin(current, present, "ET_TR3_O2_STEP_CLEAR_BRIDGE")
            absent_defined = nm_symbols(absent, "-gU")
            present_defined = nm_symbols(present, "-gU")
            self.assertEqual(present_defined - absent_defined, PRIVATE_BRIDGE_SYMBOLS)
            absent_undefined = nm_symbols(absent, "-gu")
            present_undefined = nm_symbols(present, "-gu")
            self.assertEqual((present_undefined - absent_undefined) & CORE_SYMBOLS,
                             CORE_SYMBOLS)


if __name__ == "__main__":
    unittest.main()
