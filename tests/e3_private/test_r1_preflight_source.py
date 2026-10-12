"""Guard the three E3-only preinvoke cuts and their feature-off boundary."""

from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
FRAME = ROOT / "src/eshkol_transformer/e3_frame.c"
HEADER = ROOT / "src/eshkol_transformer/e3_frame_internal.h"


def cut_site_errors(source: str) -> list[str]:
    sites = (
        ("static int64_t e3_dispatch(", "int64_t et_e3_private_forward_role_v1(",
         "e3_test_preinvoke_cut(frame, stage, role)"),
        ("int64_t et_e3_private_correct_v1(",
         "static int64_t e3_accumulate_dispatch(",
         "e3_test_preinvoke_cut(frame, 10, -1)"),
        ("static int64_t e3_accumulate_dispatch(",
         "int64_t et_e3_private_accumulate_v1(",
         "e3_test_preinvoke_cut(frame, 11, -1)"),
    )
    errors = []
    for begin, end, cut in sites:
        if begin not in source or end not in source:
            errors.append(f"missing function boundary: {begin}")
            continue
        body = source.split(begin, 1)[1].split(end, 1)[0]
        if body.count(cut) != 1 or body.count("et_kernel_runtime_dispatch(") != 1:
            errors.append(f"missing/duplicate preinvoke cut: {begin}")
            continue
        if body.index(cut) > body.index("et_kernel_runtime_dispatch("):
            errors.append(f"cut follows K1 invoke: {begin}")
        if not re.search(r"#ifdef ET_E3_R1_PREFLIGHT_TESTING\s+if \(e3_test_preinvoke_cut", body):
            errors.append(f"cut is not test-only: {begin}")
    return errors


class R1PreflightSourceTest(unittest.TestCase):
    def test_all_three_cuts_precede_k1_and_are_guarded(self) -> None:
        source = FRAME.read_text()
        self.assertEqual(cut_site_errors(source), [])
        self.assertIn("#ifdef ET_E3_R1_PREFLIGHT_TESTING\n/* Noninstalled", HEADER.read_text())
        build = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertEqual(build.count("-DET_E3_R1_PREFLIGHT_TESTING"), 1)
        native_flags = build.split("    package_source_flags=()\n", 1)[1].split(
            '    if [[ "${package_policy}" == e3-private-aggregate &&\n'
            '          "${e3_tuple_kind}" == zero ]]; then', 1)[0]
        self.assertRegex(
            native_flags,
            r'(?s)^    if \[\[ "\$\{package_policy\}" == e3-private-aggregate &&'
            r'(?:(?!\]\]; then).)*"\$\{e3_tuple_kind\}" == r1-source-retry'
            r'.*?\]\]; then\s+case "\$\{package_native_source\}" in'
            r'.*?"\$\{PROJECT_ROOT\}/src/eshkol_transformer/e3_frame\.c"\)'
            r'(?:(?!;;).)*if \[\[ "\$\{e3_tuple_kind\}" == r1-source-retry \]\]; then'
            r'\s+package_source_flags\+=\(-DET_E3_R1_PREFLIGHT_TESTING\)',
        )
        self.assertIn("frame->preinvoke_armed = 0;", source)
        self.assertIn("return e3_fail(E3_INTERNAL, E3_CODE_TEST_PREFLIGHT);", source)
        mutant = source.replace("e3_test_preinvoke_cut(frame, 11, -1)", "0", 1)
        self.assertTrue(cut_site_errors(mutant))

    def test_feature_off_object_has_no_test_control_symbol(self) -> None:
        cc = os.environ.get("CC", "clang")
        with tempfile.TemporaryDirectory() as temp:
            for mode in ("off", "on"):
                object_path = Path(temp) / f"frame-{mode}.o"
                command = [cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
                           "-Wpedantic", "-O2", "-I", str(ROOT / "include"),
                           "-I", str(ROOT / "native"), "-I",
                           str(ROOT / "src/eshkol_transformer")]
                if mode == "on":
                    command.append("-DET_E3_R1_PREFLIGHT_TESTING")
                subprocess.run(command + ["-c", str(FRAME), "-o", str(object_path)],
                               check=True, capture_output=True, text=True)
                symbols = subprocess.run(
                    ["nm", "-g", "--defined-only", str(object_path)],
                    check=True, capture_output=True, text=True).stdout
                present = bool(re.search(r"\bet_e3_test_preinvoke_arm_v1\b", symbols))
                self.assertEqual(present, mode == "on")


if __name__ == "__main__":
    unittest.main()
