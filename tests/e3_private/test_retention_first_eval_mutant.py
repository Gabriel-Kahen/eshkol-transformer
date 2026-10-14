"""Source-only admission controls; no compiled mutant claim."""

from pathlib import Path
import hashlib
import runpy
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "tests/e3_private"
PREFIX = "retention_first_eval_mutant_package_"
DERIVATION = runpy.run_path(str(PRIVATE / "retention_first_eval_mutant_source.py"))
BASE, MUTANT, transform = (DERIVATION[name] for name in ("BASE", "MUTANT", "transform"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_native(source: str) -> None:
    required = (
        '#include "../../src/eshkol_transformer/e3_frame.c"',
        "defined(ET_E3_RETENTION_TESTING) &&",
        "defined(ET_E3_FIRST_EVAL_MUTATION_TESTING)",
        "if (first_eval_mutation_used || selector < 0 || selector > 2)",
        "owner *model = admit(pointer, OWNER, 0)",
        "model->r.state != 1 || model->active",
        "metadata.state != ET_F32_GRADIENT_ABSENT",
        "metadata.state != ET_F32_GRADIENT_PRESENT",
        "et_f32_tensor_copy_bits_to_v1(gradient, words, count, &error)",
        "et_e3_test_seed_gradient_v1(pointer)",
        "et_f32_parameter_test_set_gradient_bits_v1(model->p[0], words, count)",
        "et_f32_parameter_test_set_metadata_v1(model->p[0]",
        "for (size_t i = 0; i < count; ++i)",
        "et_e3_test_model_word_v1(pointer, 0, 1, (int64_t)i)",
        "et_e3_test_gradient_metadata_v1(pointer, 0, 1)",
        "first_eval_mutation_used = 1",
    )
    if any(source.count(item) != 1 for item in required):
        raise ValueError("first-call native guard or readback missing")
    if not (source.index(required[1]) < source.index("int64_t et_e3_test_first_eval_corrupt_v1")
            < source.index(required[3]) < source.index(required[4])
            < source.index(required[8]) < source.index(required[10])
            < source.index(required[12]) < source.index(required[14])):
        raise ValueError("native denial, write, full readback or one-shot order changed")


class FirstEvaluationMutantTuple(unittest.TestCase):
    def test_exact_driver_inverse_and_first_call_boundary(self) -> None:
        base, mutant = BASE.read_text(), MUTANT.read_text()
        self.assertEqual(sha(BASE),
                         "5c2d5dcd28219e5977e574a75c983366b7fcd243e6ba100a5c9b0eee1ed9c97d")
        self.assertEqual(transform(base), mutant)
        self.assertEqual(transform(mutant, reverse=True), base)
        measured = mutant.index("(- (e3-test-region-used) region-before)")
        precheck = mutant.index("'pre-mutation-model-image")
        apply = mutant.index("'mutation-applied")
        readback = mutant.index("'mutation-readback")
        compare = mutant.index("'success-model-image")
        self.assertTrue(measured < precheck < apply < readback < compare)
        self.assertEqual(mutant.count("(vector-set! e3-test-retention-first-mutation 0 mutation)"), 1)
        self.assertEqual(mutant.count("(vector-set! e3-test-retention-first-mutation 0 -1)"), 1)
        for missing in ("'pre-mutation-model-image", "'mutation-readback",
                        "(vector-set! e3-test-retention-first-mutation 0 -1)"):
            with self.subTest(missing=missing):
                self.assertNotEqual(transform(base), mutant.replace(missing, "removed", 1))

    def test_normal_inputs_unchanged_and_manifests_exact(self) -> None:
        self.assertEqual(sha(ROOT / "src/eshkol_transformer/e3_frame.c"),
                         "8d238ed79b0b14d74b8ed454e328bcc22da46e766e4d70a59482763ef18b8857")
        digest = hashlib.sha256()
        originals = sorted(PRIVATE.glob("retention_package_*.txt"))
        self.assertEqual(len(originals), 11)
        for path in originals:
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
            copied = PRIVATE / (PREFIX + path.name.removeprefix("retention_package_"))
            expected = path.read_text()
            if path.name.endswith("_source_closure.txt") and not path.name.endswith("_native_source_closure.txt"):
                expected = expected.replace("tests/e3_private/retention_driver_root.esk",
                                            "tests/e3_private/retention_first_eval_mutant_driver_root.esk")
                expected = expected.replace("tests/e3_private/retention_driver.esk",
                                            "tests/e3_private/retention_first_eval_mutant_driver.esk")
            if path.name.endswith("_native_source_closure.txt"):
                expected = expected.replace("tests/e3_private/retention_bridge.c",
                                            "tests/e3_private/retention_first_eval_mutant_bridge.c")
                expected = expected.replace("src/eshkol_transformer/e3_frame.c\n",
                                            "tests/e3_private/retention_first_eval_frame.c\n"
                                            "src/eshkol_transformer/e3_frame.c\n", 1)
            self.assertEqual(copied.read_text(), expected, copied.name)
        self.assertEqual(digest.hexdigest(),
                         "9e11fe618c284dcc20c0c26f1b65d808345511fe303bb2fb115377d963cedaa7")
        self.assertEqual((PRIVATE / "retention_first_eval_mutant_bridge.c").read_bytes(),
                         (PRIVATE / "retention_bridge.c").read_bytes())

    def test_private_native_one_shot_and_feature_off(self) -> None:
        source = (PRIVATE / "retention_first_eval_frame.c").read_text()
        audit_native(source)
        for old, new in (
            ("first_eval_mutation_used || ", ""),
            ("selector > 2", "selector > 3"),
            ("model->r.state != 1 || model->active", "model->r.state != 1"),
            ("metadata.state != ET_F32_GRADIENT_ABSENT", "0"),
            ("metadata.state != ET_F32_GRADIENT_PRESENT", "0"),
            ("et_f32_tensor_copy_bits_to_v1(gradient, words, count, &error)", "0"),
            ("for (size_t i = 0; i < count; ++i)", "for (size_t i = 0; i < 1; ++i)"),
            ("defined(ET_E3_FIRST_EVAL_MUTATION_TESTING)", "defined(ET_E3_TESTING)"),
        ):
            with self.subTest(old=old), self.assertRaises(ValueError):
                audit_native(source.replace(old, new, 1))
        self.assertNotIn("et_e3_test_first_eval_corrupt_v1",
                         (ROOT / "src/eshkol_transformer/e3_frame.c").read_text())
        for manifest in ("defined_symbols", "public_exports", "public_strings"):
            self.assertNotIn("first_eval_corrupt",
                             (PRIVATE / f"{PREFIX}{manifest}.txt").read_text())

    def test_actual_tuple_policy_rejects_mixed_inputs(self) -> None:
        prefix = PRIVATE / "retention_first_eval_mutant_package"
        entry = PRIVATE / "retention_first_eval_mutant_driver_root.esk"
        bridge = PRIVATE / "retention_first_eval_mutant_bridge.c"
        script = '''set -euo pipefail
die() { printf '%s\\n' "$*" >&2; exit 77; }
PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$3
raw_package_renames=$4
raw_public_exports=$5
raw_include_dirs=("$PROJECT_ROOT/internal/p1/lib" "$PROJECT_ROOT/internal/c1/lib"
  "$PROJECT_ROOT/internal/t1/lib" "$PROJECT_ROOT/src"
  "$PROJECT_ROOT/internal/d2/lib" "$PROJECT_ROOT/internal/e3/lib")
source "$PROJECT_ROOT/scripts/e3-private-package-policy.sh"
printf '%s\\n' "$e3_tuple_kind"
'''
        args = [str(ROOT), str(entry), str(bridge),
                str(prefix) + "_private_renames.txt",
                str(prefix) + "_public_exports.txt"]
        def select(argv: list[str]) -> subprocess.CompletedProcess:
            return subprocess.run(["/usr/bin/bash", "-c", script, "policy", *argv],
                                  capture_output=True, text=True)
        valid = select(args)
        self.assertEqual((valid.returncode, valid.stdout),
                         (0, "retention-first-eval-mutant\n"), valid.stderr)
        for index, foreign in ((1, PRIVATE / "retention_driver_root.esk"),
                               (2, PRIVATE / "retention_bridge.c"),
                               (3, PRIVATE / "retention_package_private_renames.txt")):
            wrong = args.copy()
            wrong[index] = str(foreign)
            with self.subTest(index=index):
                self.assertNotEqual(select(wrong).returncode, 0)

    def test_runtime_and_builder_select_only_the_private_mutant(self) -> None:
        runtime = (PRIVATE / "retention_first_eval_mutant_runtime.esk").read_text()
        self.assertIn("(vector 'retention 'prepare horizon present? mutation)", runtime)
        self.assertIn("FIRST-EVAL-MUTANT-SURVIVED", runtime)
        for mode in ("absent-to-present", "seeded-bit", "seeded-metadata"):
            self.assertIn(f'"{mode}"', runtime)
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn('package_native_sources[6]="${PROJECT_ROOT}/tests/e3_private/retention_first_eval_frame.c"', builder)
        self.assertIn("-DET_E3_FIRST_EVAL_MUTATION_TESTING", builder)
        self.assertIn("e3_required_private+=(et_e3_test_first_eval_corrupt_v1)", builder)
        self.assertIn("E3_FIRST_EVAL_MUTANT_SANITIZE", builder)
        self.assertIn("retention-first-eval-mutant|retention-overflow", builder)

    def test_native_executable_control_is_authentic_and_feature_scoped(self) -> None:
        control = (PRIVATE / "retention_first_eval_native_control.c").read_text()
        runner = (ROOT / "scripts/test-e3-first-eval-native-control.sh").read_text()
        required = (
            '#include "retention_first_eval_frame.c"',
            "et_m3t_private_owner_create_v1(initial)",
            "et_m3t_private_owner_bind_v1(model",
            "et_m3t_private_owner_initialize_v1(model",
            "et_m3t_private_owner_seal_v1(model)",
            "et_m3t_private_owner_abort_v1(stale)",
            "et_e3_test_model_word_count_v1(",
            "et_e3_test_model_word_v1(",
            "et_e3_test_gradient_metadata_v1(model",
            "denied_unchanged(model, stale, selector)",
            "denied_unchanged(model, &checks, selector)",
            "denied_unchanged(model, model->original, selector)",
            "denied_unchanged(model, model, -1)",
            "denied_unchanged(model, model, 3)",
            "model->active = model",
            "model->r.state = 0",
            "ET_F32_GRADIENT_PRESENT, 2",
            "CHECK(first_eval_mutation_used == 0)",
            "CHECK(first_eval_mutation_used == 1)",
            "memcmp(&before, &after, sizeof(before)) != 0",
            "memcmp(&after, &second, sizeof(after)) == 0",
        )
        for item in required:
            self.assertIn(item, control)
        self.assertLess(control.index("denied_unchanged(model, stale"),
                        control.index("CHECK(et_e3_test_first_eval_corrupt_v1(model, selector) == 0)"))
        self.assertLess(control.index("CHECK(et_e3_test_first_eval_corrupt_v1(model, selector) == 0)"),
                        control.rindex("CHECK(et_e3_test_first_eval_corrupt_v1(model, selector) == -1)"))
        self.assertIn("-DET_E3_RETENTION_TESTING", runner)
        self.assertIn("-DET_E3_FIRST_EVAL_MUTATION_TESTING", runner)
        self.assertIn('"${out}/feature-off.o"', runner)
        self.assertIn("nm -a --defined-only", runner)
        self.assertIn('"${out}/native-${mode}" "${selector}"', runner)
        self.assertIn("detect_leaks=1:halt_on_error=1", runner)
        self.assertIn('cmp "${out}/normal-${selector}.stdout"', runner)

    def test_native_denials_restore_image_and_feature_off_requires_nm(self) -> None:
        control = (PRIVATE / "retention_first_eval_native_control.c").read_text()
        owner_state = control.split("static void denied_wrong_owner_state", 1)[1].split(
            "static void denied_wrong_gradient_state", 1)[0]
        self.assertLess(owner_state.index("snapshot(model, &before)"),
                        owner_state.index("model->r.state = 0"))
        self.assertLess(owner_state.index("model->r.state = 0"),
                        owner_state.index("et_e3_test_first_eval_corrupt_v1(model, selector)"))
        self.assertLess(owner_state.index("CHECK(model->r.state == 0 && first_eval_mutation_used == 0)"),
                        owner_state.index("model->r.state = 1"))
        self.assertLess(owner_state.index("model->r.state = 1"),
                        owner_state.index("snapshot(model, &after)"))
        gradient_state = control.split("static void denied_wrong_gradient_state", 1)[1].split(
            "int main", 1)[0]
        for item in ("selector == 0 ? ET_F32_GRADIENT_PRESENT",
                     ": ET_F32_GRADIENT_ABSENT", "denied_unchanged(model, model, selector)",
                     "first_eval_mutation_used == 0", "snapshot(model, &after)"):
            if item == "first_eval_mutation_used == 0":
                self.assertIn(item, control.split("static void denied_unchanged", 1)[1])
            else:
                self.assertIn(item, gradient_state)
        self.assertLess(gradient_state.index("snapshot(model, &before)"),
                        gradient_state.index("et_f32_parameter_test_set_metadata_v1(model->p[0], state"))
        self.assertIn("(uint64_t)before.metadata[0][1]", gradient_state)
        self.assertIn("(uint32_t)before.metadata[0][2]", gradient_state)
        self.assertIn("wrong.metadata[0][1] == before.metadata[0][1]", gradient_state)
        self.assertIn("wrong.metadata[0][2] == before.metadata[0][2]", gradient_state)
        self.assertLess(gradient_state.index("denied_unchanged(model, model, selector)"),
                        gradient_state.index("(uint32_t)before.metadata[0][0]"))
        self.assertIn("denied_wrong_owner_state(model, selector)", control)
        self.assertIn("denied_wrong_gradient_state(model, selector)", control)
        runner = ROOT / "scripts/test-e3-first-eval-native-control.sh"
        raw = runner.read_text()
        for item in ('feature-off.nm.stdout', 'feature-off.nm.stderr',
                     'feature-off.nm.exit', '(( nm_status == 0 ))',
                     'feature-off.grep.exit', '(( grep_status == 1 ))'):
            self.assertIn(item, raw)
        self.assertEqual(subprocess.run(["/usr/bin/bash", "-n", str(runner)],
                                        capture_output=True, check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
