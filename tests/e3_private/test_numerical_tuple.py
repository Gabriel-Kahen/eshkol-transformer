"""Static exact-tuple and compiled-mutant boundary checks."""

from pathlib import Path
import copy
import shutil
import subprocess
import tempfile
import unittest

from tests.e3_private.parse_numerical_observation import ROLE_NAMES, parse
from tests.e3_private.verify_numerical_result import verify_transcript_relations


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "native/e3_private_package"
NUMERICAL = ROOT / "tests/e3_private/numerical_package"


def lines(prefix: Path, suffix: str) -> list[str]:
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


class NumericalTupleTest(unittest.TestCase):
    def test_compiled_packed_rows_and_roles_require_exact_replay(self) -> None:
        names = ("packed-single", "packed-sharded", "packed-suffix",
                 "unpacked-three", "unpacked-pairs")
        rows = [dict(inputs=[1, 2], targets=[2, 3], mask=[True, True],
                     logits_f32=["00000000"], ce_f32=["3f800000"],
                     reduction_f32=["40000000"], correct_f32="00000000")
                for _ in range(3)]
        observed = {name: {"batches": copy.deepcopy(rows[1:] if name == "packed-suffix"
                                                 else rows)} for name in names}
        roles = {name: tuple((role, "1,2,4", "00000000")
                             for _ in range(2 if name == "packed-suffix" else 3)
                             for role in ROLE_NAMES) for name in names}
        verify_transcript_relations(observed, roles)
        changed = copy.deepcopy(observed)
        changed["packed-suffix"]["batches"][0]["ce_f32"] = ["40000000"]
        with self.assertRaisesRegex(ValueError, "suffix row words"):
            verify_transcript_relations(changed, roles)
        changed_roles = dict(roles)
        changed_roles["packed-sharded"] = tuple(
            ("ET", "1,2,4", "3f800000") if index == 0 else row
            for index, row in enumerate(roles["packed-sharded"])
        )
        with self.assertRaisesRegex(ValueError, "repartition rows or roles"):
            verify_transcript_relations(observed, changed_roles)

    def test_private_generated_source_stage_is_stable_and_fresh(self) -> None:
        helper = ROOT / "scripts/e3-generated-source-stage.sh"
        with tempfile.TemporaryDirectory(prefix="e3-numerical-gate-") as temporary:
            gate = Path(temporary)
            stages = [gate / ".e3-numerical.ABC123",
                      gate / ".e3-numerical.XYZ789"]
            for stage in stages:
                stage.mkdir()

            def select(object_path: Path) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["bash", "-c", 'source "$1"; e3_numerical_generated_stage "$2"',
                     "bash", str(helper), str(object_path)],
                    capture_output=True, text=True, check=False,
                )

            expected = str(gate / ".e3-numerical-generated") + "\n"
            for stage in stages:
                selected = select(stage / "e3_numerical_package.o")
                self.assertEqual((selected.returncode, selected.stdout),
                                 (0, expected), selected.stderr)
            self.assertNotEqual(select(stages[0] / "other.o").returncode, 0)
            alias = gate / ".e3-numerical.ALIAS1"
            alias.symlink_to(stages[0], target_is_directory=True)
            self.assertNotEqual(select(alias / "e3_numerical_package.o").returncode, 0)
            generated = gate / ".e3-numerical-generated"
            generated.symlink_to(stages[0], target_is_directory=True)
            self.assertNotEqual(select(stages[0] / "e3_numerical_package.o").returncode, 0)

    def test_real_asan_common_cell_is_private_after_relocatable_link(self) -> None:
        # Exercise actual Clang instrumentation and binutils, not a stubbed nm.
        required = ("clang", "clang++", "nm", "objcopy")
        if any(shutil.which(tool) is None for tool in required):
            self.skipTest("Clang/binutils sanitizer object tools unavailable")
        with tempfile.TemporaryDirectory(prefix="e3-asan-common-") as temporary:
            directory = Path(temporary)
            source = directory / "probe.c"
            source.write_text("int fixture_global;\n"
                              "int fixture_public(void) { return fixture_global; }\n")
            def run(*command: str) -> str:
                return subprocess.check_output(command, cwd=directory,
                                               text=True, stderr=subprocess.STDOUT)
            run("clang", "-fPIC", "-fsanitize=address,undefined", "-c",
                str(source), "-o", "probe.o")
            symbols = run("nm", "-g", "--defined-only", "--format=posix", "probe.o")
            self.assertIn("___asan_globals_registered C ", symbols)
            private_names = [line.split()[0] for line in symbols.splitlines()
                             if line.split()[0] != "fixture_public"]
            (directory / "private.txt").write_text("\n".join(private_names) + "\n")
            # Reproduce the observed failure: localization alone leaves COMMON.
            run("objcopy", "--localize-symbols=private.txt", "probe.o", "old.o")
            self.assertIn("___asan_globals_registered C ",
                          run("nm", "-g", "--defined-only", "--format=posix", "old.o"))
            run("clang++", "-r", "-Wl,-d", "probe.o", "-o", "defined.o")
            run("objcopy", "--localize-symbols=private.txt", "defined.o", "fixed.o")
            final = run("nm", "-g", "--defined-only", "--format=posix", "fixed.o")
            self.assertEqual([line.split()[0] for line in final.splitlines()],
                             ["fixture_public"])
            all_symbols = run("nm", "--defined-only", "--format=posix", "fixed.o")
            self.assertIn("___asan_globals_registered b ", all_symbols)
            imports = run("nm", "-u", "--format=posix", "fixed.o")
            self.assertIn("__asan_init U", imports)
            builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
            self.assertIn("e3_private_test_reloc_flags=()", builder)
            self.assertIn("e3_private_test_reloc_flags=(-Wl,-d)", builder)
            self.assertIn('"${e1b_cxx}" -r "${e3_private_test_reloc_flags[@]}"', builder)

    def test_real_sanitized_optimization_import_delta(self) -> None:
        added = lines(NUMERICAL, "sanitized_added_undefined_symbols")
        absent = lines(NUMERICAL, "sanitized_absent_undefined_symbols")
        self.assertEqual(added, ["bcmp", "puts"])
        self.assertEqual(absent, ["eshkol_ad_seed_flag", "eshkol_runtime_fatal",
                                  "eshkol_taylor_lift_ad_node", "memcmp"])
        if any(shutil.which(tool) is None for tool in ("clang", "nm")):
            self.skipTest("Clang/binutils sanitizer object tools unavailable")
        with tempfile.TemporaryDirectory(prefix="e3-sanitizer-imports-") as temporary:
            directory = Path(temporary)
            source = directory / "probe.c"
            source.write_text(
                '#include <stdio.h>\n#include <string.h>\n'
                'int equal(const void *a, const void *b) '
                '{ return memcmp(a, b, 32) == 0; }\n'
                'void print(void) { printf("fixture\\n"); }\n'
            )
            object_file = directory / "probe.o"
            subprocess.run(
                ["clang", "-O1", "-g", "-fsanitize=address,undefined",
                 "-fno-omit-frame-pointer", "-c", str(source), "-o", str(object_file)],
                check=True, capture_output=True, text=True,
            )
            imports = subprocess.run(
                ["nm", "-u", "--format=posix", str(object_file)],
                check=True, capture_output=True, text=True,
            ).stdout
            names = {line.split()[0] for line in imports.splitlines()}
            self.assertTrue(set(added) <= names)
            self.assertFalse({"memcmp", "printf"} & names)
            self.assertIn("__asan_init", names)

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

    def test_native_provider_mode_tracks_both_test_sanitizers(self) -> None:
        policy = ROOT / "scripts/e3-private-native-inputs.sh"
        with tempfile.TemporaryDirectory(prefix="e3-native-mode-") as temporary:
            sandbox = Path(temporary)
            scripts = sandbox / "scripts"
            scripts.mkdir()
            objects = ["n2/n2_primitives_provider.o",
                       "n3k/n3k_primitives_provider.o",
                       "a2/a2_attention_provider.o"]
            (sandbox / "tuple_native_objects.txt").write_text(
                "\n".join(objects) + "\n"
            )
            for provider in ("n2", "n3k", "a2"):
                stub = scripts / f"build-{provider}.sh"
                stub.write_text(
                    "#!/usr/bin/env bash\n"
                    f"printf '%s\\n' \"$2\" >'{sandbox}/mode-{provider}'\n"
                )
                stub.chmod(0o755)
            shell = '''
set -euo pipefail
PROJECT_ROOT=$1
e1b_tmp=$2
e3_prefix=$3
e3_private_test_sanitize=$4
package_policy=e3-private-aggregate
e1b_clean_toolchain_env=()
package_native_objects=()
package_native_depfiles=()
die() { printf '%s\\n' "$*" >&2; exit 77; }
source "$5"
'''
            for value, expected in (("0", "normal"), ("1", "sanitize")):
                run = subprocess.run(
                    ["bash", "-c", shell, "bash", str(sandbox), str(sandbox),
                     str(sandbox / "tuple"), value, str(policy)],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                for provider in ("n2", "n3k", "a2"):
                    self.assertEqual((sandbox / f"mode-{provider}").read_text(),
                                     expected + "\n")

    def test_incomplete_compiled_stdout_cannot_become_observation(self) -> None:
        with self.assertRaisesRegex(ValueError, "did not complete"):
            parse(b"E3ROW\t0\t3,197\t197,0\t1,1\n")
        with self.assertRaisesRegex(ValueError, "missing, duplicate, or reordered"):
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
                      "E3RESULT\t1065353216,1073741824,1077936128,0,6,3\t"
                      + ",".join(["0"] * 200) + "\t" + ",".join(["1"] * 200),
                      "E3-NUMERICAL-RUNTIME-PASS"))
        raw = ("\n".join(lines) + "\n").encode("ascii")
        payload, roles, result = parse(raw)
        self.assertEqual(payload["batches"][0]["inputs"], [3, 197])
        self.assertEqual(payload["batches"][2]["targets"], [197, 0])
        self.assertEqual(payload["batches"][1]["ce_f32"],
                         ["3f800000", "40000000"])
        self.assertEqual(payload["final"]["metrics_f32"][0], "3f800000")
        self.assertEqual(len(roles.splitlines()), 63)
        self.assertEqual(result["outputs"][4:], [6, 3])
        self.assertEqual(len(bytes.fromhex(result["cursor_before"])), 200)
        for malformed in (
            lines[:1] + lines[2:],                         # missing role
            lines[:2] + lines[1:],                         # duplicate role
            lines[:2] + [lines[3], lines[2]] + lines[4:], # reordered roles
            lines[:-2] + lines[-1:],                       # missing boxed result
            lines[:-1] + [lines[-2], lines[-1]],           # duplicate boxed result
            lines[:-2] + [lines[-2].replace(",0,6,3", ",1,6,3")] + lines[-1:],
        ):
            with self.assertRaises(ValueError):
                parse(("\n".join(malformed) + "\n").encode("ascii"))
        suffix = lines[25:75] + lines[-3:]
        suffix = ["\t".join([parts[0], str(int(parts[1]) - 1), *parts[2:]])
                  if line.startswith(("E3ROW\t", "E3ROLE\t", "E3CE\t",
                                      "E3BATCH\t", "E3STATE\t")) else line
                  for line in suffix for parts in [line.split("\t")]]
        suffix_payload, suffix_roles, _ = parse(
            ("\n".join(suffix) + "\n").encode("ascii"), 2)
        self.assertEqual((len(suffix_payload["batches"]), len(suffix_roles.splitlines())),
                         (2, 42))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse(("\n".join(lines[:-1] + [lines[0], lines[-1]]) + "\n").encode("ascii"))


if __name__ == "__main__":
    unittest.main()
