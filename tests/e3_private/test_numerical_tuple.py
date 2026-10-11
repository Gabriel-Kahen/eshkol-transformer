"""Static exact-tuple and compiled-mutant boundary checks."""

from pathlib import Path
import math
import re
import shutil
import subprocess
import tempfile
import unittest

from tests.e3_private.parse_numerical_observation import ROLE_NAMES, parse
from tests.c2.test_handler_order import parse as parse_source
from tests.e3_reference.corpus import materialize
from tests.e3_reference.transcript import (
    TranscriptError, f32_bits, validate_observed_case,
)
from tests.e3_metrics.reference_probe import accumulate, divide, finalize, reduce_bool


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "native/e3_private_package"
NUMERICAL = ROOT / "tests/e3_private/numerical_package"


def lines(prefix: Path, suffix: str) -> list[str]:
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


def frame_lines(batch_count: int) -> list[str]:
    records = []
    for ordinal in range(batch_count):
        records.append(f"E3ROW\t{ordinal}\t3,197\t197,0\t1,1")
        for name in ROLE_NAMES:
            shape = "1,2,256" if name == "Z" else "1"
            words = ",".join(["3f800000"] * (512 if name == "Z" else 1))
            records.append(f"E3ROLE\t{ordinal}\t{name}\t{shape}\t{words}")
        records.extend((
            f"E3CE\t{ordinal}\t3f800000,40000000",
            f"E3BATCH\t{ordinal}\t40400000,40000000,3fc00000,00000000\t2",
            f"E3STATE\t{ordinal}\t40400000,40000000,00000000\t{2 * (ordinal + 1)},{ordinal + 1}",
        ))
    records.extend((f"E3FINAL\t3f800000,40000000,40400000,00000000\t{2 * batch_count},{batch_count}",
                    "E3-NUMERICAL-RUNTIME-PASS"))
    return records


def frame_bytes(records: list[str]) -> bytes:
    return ("\n".join(records) + "\n").encode("ascii")


class NumericalTupleTest(unittest.TestCase):
    def test_runtime_drivers_parse_completely_and_reject_extra_close(self) -> None:
        # A stray close previously stopped the supported gate before runtime.
        for name in ("numerical_runtime.esk", "correctness_runtime.esk"):
            with self.subTest(driver=name):
                source = (ROOT / "tests/e3_private" / name).read_text()
                forms = parse_source(source)
                arguments = [node for node in forms
                             if isinstance(node, list) and len(node) > 1
                             and node[:2] == ["check", ["quote", "arguments"]]]
                self.assertEqual(len(arguments), 1)
                self.assertEqual(forms[-2:],
                                 [["display", '"E3-NUMERICAL-RUNTIME-PASS"'],
                                  ["newline"]])
                with self.assertRaisesRegex(ValueError, "unexpected close"):
                    parse_source(source + ")")
                with self.assertRaisesRegex(ValueError, "unclosed list"):
                    parse_source("(" + source)

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
                                  "eshkol_taylor_lift_ad_node", "memcmp",
                                  "memcpy", "memmove", "memset"])
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
                'void copy(void *d, const void *s, size_t n) { memcpy(d, s, n); }\n'
                'void move(void *d, const void *s, size_t n) { memmove(d, s, n); }\n'
                'void fill(void *d, int v, size_t n) { memset(d, v, n); }\n'
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
            self.assertFalse({"memcmp", "printf", "memcpy", "memmove", "memset"} & names)
            self.assertTrue({"__asan_memcpy", "__asan_memmove", "__asan_memset"} <= names)
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
        self.assertIn("-DET_E3_NUMERICAL_SECOND_NORMALIZATION", builder)
        self.assertIn("wrong-vocab-axis) package_native_cflags+=(-DET_E3_NUMERICAL_WRONG_VOCAB_AXIS)", builder)
        self.assertIn("second-normalization", (ROOT / "scripts/test-e3-numerical-transcript.sh").read_text())
        self.assertIn("#if defined(ET_E3_NUMERICAL_TESTING)", frame)
        self.assertIn("#ifdef ET_E3_NUMERICAL_WRONG_TARGET", frame)
        self.assertIn("#ifdef ET_E3_NUMERICAL_MEAN_OF_MEANS", metrics)
        self.assertIn("#ifdef ET_E3_NUMERICAL_SECOND_NORMALIZATION", metrics)
        self.assertIn("#ifdef ET_E3_NUMERICAL_WRONG_VOCAB_AXIS", metrics)
        self.assertNotIn("E3ROW", (ROOT / "tests/e3_private/driver.esk").read_text())

    def test_wrong_vocab_axis_macro_requires_private_numerical_tuple(self) -> None:
        if shutil.which("clang") is None:
            self.skipTest("Clang unavailable")
        source = ROOT / "native/e3_evaluation_metrics_provider.c"
        base = ["clang", "-std=c11", "-fsyntax-only", "-I", str(ROOT / "include"),
                "-I", str(ROOT / "src")]
        forbidden = subprocess.run(
            base + ["-DET_E3_NUMERICAL_WRONG_VOCAB_AXIS", str(source)],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(forbidden.returncode, 0)
        self.assertIn("requires the private numerical test tuple", forbidden.stderr)
        admitted = subprocess.run(
            base + ["-DET_E3_NUMERICAL_TESTING", "-DET_E3_NUMERICAL_WRONG_VOCAB_AXIS",
                    str(source)], capture_output=True, text=True, check=False,
        )
        self.assertEqual(admitted.returncode, 0, admitted.stderr)
        selector = (ROOT / "scripts/build-e3-numerical-test.sh").read_text()
        runner = (ROOT / "scripts/test-e3-correctness-cases.sh").read_text()
        self.assertIn("wrong-vocab-axis|physical-capacity|sanitize", selector)
        self.assertIn('wrong_vocab_axis_rejection.py" run', runner)
        self.assertIn('wrong_vocab_axis_rejection.py" check', runner)
        self.assertIn('verify-wrong-vocab-axis-all-correct.exit', runner)
        self.assertIn('"${e3_tuple_kind}" == numerical',
                      (ROOT / "scripts/build-e1b-consumer.sh").read_text())

    def test_wrong_vocab_axis_selector_rejects_mixed_and_near_miss_tuples(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        selector = builder.split("e3_numerical_mutant=${E3_NUMERICAL_MUTANT:-normal}", 1)[1]
        selector = "e3_numerical_mutant=${E3_NUMERICAL_MUTANT:-normal}" + selector.split(
            "require_command awk", 1)[0]
        shell = ("set -euo pipefail\n"
                 "package_policy=$1\ne3_tuple_kind=$2\n"
                 "die() { printf '%s\\n' \"$*\" >&2; exit 77; }\n"
                 + selector + "printf '%s\\n' \"$e3_numerical_mutant\"\n")
        def select(policy: str, kind: str, value: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["bash", "-c", shell, "bash", policy, kind],
                env={"E3_NUMERICAL_MUTANT": value}, capture_output=True,
                text=True, check=False,
            )
        accepted = select("e3-private-aggregate", "numerical", "wrong-vocab-axis")
        self.assertEqual((accepted.returncode, accepted.stdout),
                         (0, "wrong-vocab-axis\n"), accepted.stderr)
        for policy, kind, value in (("e3-private-aggregate", "base", "wrong-vocab-axis"),
                                    ("e3-private-aggregate", "mode-graph", "wrong-vocab-axis"),
                                    ("other", "numerical", "wrong-vocab-axis"),
                                    ("e3-private-aggregate", "numerical", "wrong-vocab-axes")):
            with self.subTest(policy=policy, kind=kind, value=value):
                self.assertEqual(select(policy, kind, value).returncode, 77)

    def test_physical_capacity_selector_and_padded_witness(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        selector = builder.split("e3_numerical_mutant=${E3_NUMERICAL_MUTANT:-normal}", 1)[1]
        selector = "e3_numerical_mutant=${E3_NUMERICAL_MUTANT:-normal}" + selector.split(
            "require_command awk", 1)[0]
        shell = ("set -euo pipefail\npackage_policy=$1\ne3_tuple_kind=$2\n"
                 "die() { printf '%s\\n' \"$*\" >&2; exit 77; }\n"
                 + selector + "printf '%s\\n' \"$e3_numerical_mutant\"\n")
        def select(policy: str, kind: str, value: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["bash", "-c", shell, "bash", policy, kind],
                env={"E3_NUMERICAL_MUTANT": value}, capture_output=True,
                text=True, check=False,
            )
        accepted = select("e3-private-aggregate", "numerical", "physical-capacity")
        self.assertEqual((accepted.returncode, accepted.stdout),
                         (0, "physical-capacity\n"), accepted.stderr)
        for policy, kind, value in (("e3-private-aggregate", "base", "physical-capacity"),
                                    ("e3-private-aggregate", "mode-graph", "physical-capacity"),
                                    ("other", "numerical", "physical-capacity"),
                                    ("e3-private-aggregate", "numerical", "physical-capacities")):
            with self.subTest(policy=policy, kind=kind, value=value):
                self.assertEqual(select(policy, kind, value).returncode, 77)

        with tempfile.TemporaryDirectory(prefix="e3-m2-physical-witness-") as temporary:
            cases = {case["name"]: case for case in materialize(Path(temporary) / "corpus")["cases"]}
        masks = [row["mask"] for row in cases["packed-single"]["batches"]]
        self.assertEqual(masks, [[True, True], [True, True], [True, False]])
        self.assertEqual(sum(sum(mask) for mask in masks), 5)
        self.assertEqual(sum(sum((True, True)) for _ in masks), 6)
        genuine = reduce_bool((0x3f800000, 0x40000000), tuple(masks[-1]))
        mutant = reduce_bool((0x3f800000, 0x40000000), (True, True))
        self.assertEqual((genuine[1], mutant[1]), (0x3f800000, 0x40000000))

        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        self.assertIn("(unsigned)frame->mask[0], (unsigned)frame->mask[1]", frame)
        for begin, end in (("int64_t et_e3_private_reduce_v1", "int64_t et_e3_private_correct_v1"),
                           ("int64_t et_e3_private_correct_v1", "static int64_t e3_accumulate_dispatch")):
            body = frame.split(begin, 1)[1].split(end, 1)[0]
            self.assertIn("frame->mask[0] == 1u && frame->mask[1] == 0u", body)
            self.assertIn("metric_mask, sizeof(frame->mask)", body)
            self.assertIsNone(re.search(r"frame->mask\[[01]\]\s*=(?!=)", body))
        runner = (ROOT / "scripts/test-e3-numerical-transcript.sh").read_text()
        self.assertIn("batches[2].reduction_f32", runner)
        self.assertIn("physical-capacity", (ROOT / "scripts/build-e3-numerical-test.sh").read_text())

    def test_physical_capacity_macro_requires_private_numerical_tuple(self) -> None:
        if shutil.which("clang") is None:
            self.skipTest("Clang unavailable")
        source = ROOT / "src/eshkol_transformer/e3_frame.c"
        base = ["clang", "-std=c11", "-Werror", "-fsyntax-only",
                "-I", str(ROOT / "include"), "-I", str(ROOT / "src"),
                "-I", str(ROOT / "src/eshkol_transformer")]
        for flags, success in (([], True),
                               (["-DET_E3_NUMERICAL_PHYSICAL_CAPACITY"], False),
                               (["-DET_E3_NUMERICAL_TESTING",
                                 "-DET_E3_NUMERICAL_PHYSICAL_CAPACITY"], True)):
            with self.subTest(flags=flags):
                run = subprocess.run(base + flags + [str(source)],
                                     capture_output=True, text=True, check=False)
                self.assertEqual(run.returncode == 0, success, run.stderr)
                if not success:
                    self.assertIn("requires the private numerical test tuple", run.stderr)

    def test_physical_capacity_synthetic_comparator_fails_at_reduction(self) -> None:
        # Comparator sensitivity only: compiled M3/D2 evidence remains open.
        ce = f32_bits(math.log(256.0))
        ce_bits = int(ce, 16)
        numerator, weight = reduce_bool((ce_bits, ce_bits), (True, False))
        state = accumulate((0, 0, 0, 0, 0), (numerator, weight, 0, 1))
        final = finalize(state)
        observed = {
            "batches": [{
                "ce_f32": [ce, ce], "correct_f32": "00000000",
                "counters": [1, 1], "inputs": [7, 0],
                "logits_f32": ["00000000"] * 512, "mask": [True, False],
                "ordinal": 0, "reduction_f32": [f"{numerator:08x}", f"{weight:08x}",
                                                f"{divide(numerator, weight):08x}"],
                "state_f32": [f"{word:08x}" for word in state[:3]],
                "targets": [7, 0],
            }],
            "final": {"counters": [1, 1],
                      "metrics_f32": [f"{word:08x}" for word in final]},
            "kind": "synthetic-observation",
            "profile": "e3-private-n1-t2-v256-d4-v1",
        }
        expected = ({"inputs": [7, 0], "targets": [7, 0],
                     "mask": [True, False]},)
        validate_observed_case(observed, expected)
        bad_n, bad_w = reduce_bool((ce_bits, ce_bits), (True, True))
        observed["batches"][0]["reduction_f32"] = [
            f"{bad_n:08x}", f"{bad_w:08x}", f"{divide(bad_n, bad_w):08x}",
        ]
        self.assertEqual((weight, bad_w), (0x3f800000, 0x40000000))
        with self.assertRaisesRegex(TranscriptError, r"batches\[0\]\.reduction_f32"):
            validate_observed_case(observed, expected)

    def test_physical_capacity_parser_mode_is_exact_and_default_stays_strict(self) -> None:
        raw = frame_lines(3)
        row = raw.index("E3ROW\t2\t3,197\t197,0\t1,1")
        raw[row] = "E3ROW\t2\t3,197\t197,0\t1,0"
        mutant = frame_bytes(raw)
        with self.assertRaisesRegex(ValueError, "active count differs from mask"):
            parse(mutant, 3)
        observed, roles = parse(mutant, 3, physical_capacity=True)
        self.assertEqual([batch["mask"] for batch in observed["batches"]],
                         [[True, True], [True, True], [True, False]])
        self.assertEqual(len(roles.splitlines()), 63)
        with self.assertRaisesRegex(ValueError, "requires three batches"):
            parse(mutant, 2, physical_capacity=True)
        wrong_active = frame_bytes([line.rsplit("\t", 1)[0] + "\t1"
                                    if line.startswith("E3BATCH\t2\t")
                                    else line for line in raw])
        with self.assertRaisesRegex(ValueError, "padded row must retain 10"):
            parse(wrong_active, 3, physical_capacity=True)

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
            parse(b"E3ROW\t0\t3,197\t197,0\t1,1\n", 2)
        with self.assertRaisesRegex(ValueError, "omitted"):
            parse(b"reference JSON instead of native frame\n"
                  b"E3-NUMERICAL-RUNTIME-PASS\n", 2)

    def test_parser_uses_tagged_frame_words_without_oracle_substitution(self) -> None:
        records = frame_lines(3)
        payload, roles = parse(frame_bytes(records), 3)
        self.assertEqual(payload["batches"][0]["inputs"], [3, 197])
        self.assertEqual(payload["batches"][2]["targets"], [197, 0])
        self.assertEqual(payload["batches"][1]["ce_f32"],
                         ["3f800000", "40000000"])
        self.assertEqual(payload["final"]["metrics_f32"][0], "3f800000")
        self.assertEqual(len(roles.splitlines()), 63)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse(frame_bytes(records[:-1] + [records[0], records[-1]]), 3)

    def test_two_batch_parser_requires_complete_ordered_native_records(self) -> None:
        records = frame_lines(2)
        payload, roles = parse(frame_bytes(records), 2)
        self.assertEqual([batch["ordinal"] for batch in payload["batches"]], [0, 1])
        self.assertEqual(payload["final"]["counters"], [4, 2])
        self.assertEqual(len(roles.splitlines()), 42)
        with self.assertRaisesRegex(ValueError, "omitted"):
            parse(frame_bytes(records), 3)
        for bad_count in (0, 1, 4, True):
            with self.subTest(count=bad_count), self.assertRaisesRegex(ValueError, "count"):
                parse(frame_bytes(records), bad_count)
        broken = {
            "missing-role": records[:2] + records[3:],
            "duplicate-row": records[:1] + records[:1] + records[1:],
            "swapped-roles": records[:1] + [records[2], records[1]] + records[3:],
            "wrong-ordinal": [records[0].replace("E3ROW\t0", "E3ROW\t1")] + records[1:],
            "early-final": records[:-3] + [records[-2], records[-3], records[-1]],
            "missing-final": records[:-2] + records[-1:],
            "malformed-word": records[:1] + [records[1].replace("3f800000", "zzzzzzzz")]
            + records[2:],
            "wrong-active-count": records[:23] + [records[23].replace("\t2", "\t1")]
            + records[24:],
        }
        for name, lines in broken.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse(frame_bytes(lines), 2)


if __name__ == "__main__":
    unittest.main()
