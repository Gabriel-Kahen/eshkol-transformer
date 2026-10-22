"""Host-only checks of the genuine runtime fixture and its build composition."""
from pathlib import Path
import re
import unittest

from test_source_contract import parse, walk

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/cpu4_trainer_control/runtime_test.esk"
RECIPE = ROOT / "scripts/test-cpu4-trainer-control.sh"


class RuntimeRecipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = FIXTURE.read_text()
        cls.forms = parse(cls.text)
        cls.script = RECIPE.read_text()

    def test_genuine_dependencies_in_one_aggregate(self):
        loads = [form[1] for form in self.forms if form[0] == "load"]
        self.assertEqual(len(loads), len(set(loads)))
        for name in ("t2_wave2_root.esk", "i2_wave2_extension.esk", "o2_wave2_extension.esk",
                     "d2_semantic_core.esk", "d2_dataset.esk", "c2_d2_cursor_pair_extension.esk",
                     "g3c4_model_extension.esk", "cpu4_p1_fixed_set_extension.esk",
                     "cpu4_model_binding_extension.esk", "cpu4_trainer_control_extension.esk"):
            self.assertIn(name, loads)
        self.assertLess(loads.index("d2_dataset.esk"), loads.index("c2_d2_cursor_pair_extension.esk"))
        self.assertEqual(loads[-1], "cpu4_trainer_control_extension.esk")
        for path in loads:
            self.assertTrue(any((ROOT / base / path).exists()
                                for base in ("native", "src", "lib", "internal/d2/lib")), path)
        self.assertNotIn("(extern ", self.text)
        self.assertIn("(t2-wave2-tokenizer-byte resolved)", self.text)
        self.assertNotIn("t2-wave2-tokenizer-load", self.text)
        self.assertNotIn("byte_tokenizer_v1.tsv", self.script)
        self.assertIn("(transformer-error-operation caught)", self.text)
        self.assertIn("(transformer-error-message caught)", self.text)
        for form in self.forms:
            if form[0] == "define":
                name = form[1][0] if isinstance(form[1], list) else form[1]
                self.assertFalse(name.startswith(("cpu4-", "o2-", "d2-", "g3c4-", "t2-")), name)

    def test_real_owner_calls_and_bounded_control_coverage(self):
        heads = {node[0] for form in self.forms for node in walk(form)
                 if node and isinstance(node[0], str)}
        for name in ("g3c4-model-create-seeded-internal", "cpu4-model-binding-create-internal",
                     "o2-public-optimizer-create", "t2-wave2-tokenizer-byte",
                     "d2-token-dataset-open", "d2-token-dataset-next-batch",
                     "d2-token-batch-release!", "cpu4-control-native-enter", "cpu4-control-native-leave",
                     "cpu4-trainer-create-internal", "cpu4-trainer-controls-copy-internal",
                     "cpu4-trainer-controls-restore-stage-internal",
                     "cpu4-trainer-controls-restore-precommit-internal",
                     "cpu4-trainer-controls-restore-abort-internal!",
                     "cpu4-trainer-controls-restore-commit-internal!",
                     "cpu4-model-binding-release-internal!", "module-mode-internal"):
            self.assertIn(name, heads)
        for label in ("constructor native contention", "overlapping constructor",
                      "begin native WRITE contention", "snapshot deep detach including strings/bytes",
                      "failed stage leaves same parent/no writes", "stage detaches caller input",
                      "copied stage token rejected", "committed stage repeated commit",
                      "all17 EVAL", "all17 exact TRAIN restored", "binding release repeat"):
            self.assertIn('"' + label + '"', self.text)
        self.assertNotIn("tr3-lease-", self.text)
        self.assertNotIn("m3t-public-model-create", self.text)
        self.assertNotRegex(self.text, r"\(length\s")

    def test_recipe_reuses_exact_binding_native_universe_plus_o2(self):
        self.assertIn('native/cpu4_binding_native_sources.txt', self.script)
        self.assertIn("printf '%s\\n' native/o2_optimizer.c native/o2_wave2_package_bridge.c", self.script)
        self.assertIn("native/o2_wave2_package_bridge.c) extra=(-DET_O2_NATIVE_HELPERS_ONLY)", self.script)
        self.assertIn('tests/cpu4_trainer_control/runtime_test.esk', self.script)
        self.assertIn('--strict-types --optimize 0 --no-stdlib', self.script)
        self.assertIn('-DET_CPU4_TRAINING_PRIVATE', self.script)
        self.assertIn('-DET_TR3_C_D2_RESTORE', self.script)
        self.assertIn('-ffp-contract=off -fexcess-precision=standard -fno-fast-math', self.script)
        self.assertIn('ESHKOL_ARENA_POISON', self.script)
        self.assertIn('for repeat in 1 2', self.script)
        self.assertIn('cmp "$out/run-1.stdout" "$out/run-2.stdout"', self.script)
        self.assertNotIn('flock ', self.script)
        self.assertNotRegex(self.script, r"(^|\n)\s*(make|ninja)\b")
        self.assertNotIn('ET_O2_TESTING', self.script)
        self.assertNotIn('ET_F32_TENSOR_TESTING', self.script)


if __name__ == "__main__":
    unittest.main()
