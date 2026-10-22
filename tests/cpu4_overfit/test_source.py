"""Cheap fixture/recipe guards; this never executes a training model."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/cpu4_trainer_control"))
from test_source_contract import parse, walk


class Source(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = (ROOT / "tests/cpu4_overfit/runtime_test.esk").read_text()
        cls.forms = parse(cls.fixture)
        cls.recipe = (ROOT / "scripts/test-cpu4-one-batch-overfit.sh").read_text()
        cls.bridge = (ROOT / "tests/cpu4_overfit/retention_bridge.c").read_text()

    def test_genuine_same_identity_training_and_no_fake_loss_authority(self):
        loads = [f[1] for f in self.forms if f[0] == "load"]
        self.assertEqual(len(loads), len(set(loads)))
        self.assertEqual(loads[-1], "cpu4_training_step_extension.esk")
        heads = [n[0] for f in self.forms for n in walk(f) if n and isinstance(n[0], str)]
        for required in ("g3c4-model-create-seeded-internal", "cpu4-model-binding-create-internal",
                         "o2-public-optimizer-create", "cpu4-trainer-create-internal",
                         "cpu4-training-step-internal!", "cpu4-trainer-idle-check-internal",
                         "i2-parameter-gradient-metadata-internal", "i2-checkpoint-encode",
                         "cpu4-control-native-enter", "cpu4-control-native-leave"):
            self.assertIn(required, heads)
        self.assertEqual(heads.count("cpu4-training-step-internal!"), 1)
        self.assertIn("(vector-ref result 3)", self.fixture)
        for forbidden in ("tr3-lease-", "m3t-public-model-create", "e3-evaluate", "trainer-evaluate",
                          "optimizer-step!", "native-objective", "t2-wave2-tokenizer-load"):
            self.assertNotIn(forbidden, self.fixture)
        for f in self.forms:
            if f[0] == "define":
                name = f[1][0] if isinstance(f[1], list) else f[1]
                self.assertFalse(name.startswith(("cpu4-", "o2-", "d2-", "g3c4-")), name)

    def test_exact_fixed_data_optimizer_and_update_policy(self):
        for word in ("3dcccccd", "3f000000", "3a83126f", "00000000"):
            self.assertIn('"' + word + '"', self.fixture)
        for text in ("(g3c4-model-create-seeded-internal 1729)", "(<= update 250)",
                     "(= (vector-ref record 19) (* update 4))", "(vector 0 0)",
                     "(t2-wave2-tokenizer-byte resolved)", "':maximum-total-tokens 5",
                     "(eq? model (vector-ref record 8))", "'(4 4)"):
            self.assertIn(text, self.fixture)
        self.assertIn("((3,197,3,41,7),)", self.recipe)
        self.assertIn("(eq? (vector-ref (vector-ref b 3) 7) #f)", self.fixture)

    def test_loop_names_arity_and_exact_bytewise_parameter_checks(self):
        for f in self.forms:
            for node in walk(f):
                if not node:
                    continue
                if node[0] == "if":
                    self.assertIn(len(node), (3, 4))
                if node[0] == "let" and len(node) > 1 and isinstance(node[1], str):
                    self.assertNotIn(node[1], ("values", "match", "length", "vector", "list", "map", "train"))
        self.assertIn("(d2-core-bytes=? (vector-ref before index) (vector-ref after index))", self.fixture)
        self.assertIn("(= bytes 4768)", self.fixture)
        self.assertIn("(> changed 0)", self.fixture)
        self.assertNotIn("(equal? before", self.fixture)

    def test_retention_reads_existing_counters_without_failpoints_or_training(self):
        for required in ("et_f32_test_live_counts_snapshot_v1", "et_f32_test_retired_counts_snapshot_v1",
                         "et_o2_trainer_step_clear_test_counts_snapshot_v1"):
            self.assertIn(required, self.bridge)
        for forbidden in ("fail_alloc", "optimizer_step_v1", "optimizer_zero_grad", "memcpy("):
            self.assertNotIn(forbidden, self.bridge)
        self.assertIn("memcmp(&baseline, &now", self.bridge)
        self.assertIn("67108864", self.fixture)
        self.assertIn("262144", self.fixture)

    def test_recipe_one_aggregate_exact_flags_and_post_execution_only_checker(self):
        for required in ("native/cpu4_training_native_sources.txt", "-DET_CPU4_OBJECTIVE_PRIVATE",
                         "-DET_TR3_O2_STEP_CLEAR_NATIVE -DET_O2_TESTING",
                         "-DET_O2_NATIVE_HELPERS_ONLY -DET_TR3_O2_STEP_CLEAR_BRIDGE",
                         "m3t_f32_integration.c) extra=(-DET_F32_TENSOR_TESTING)",
                         "tests/cpu4_overfit/retention_bridge.c", "--strict-types --optimize 0 --no-stdlib",
                         "-ffp-contract=off -fexcess-precision=standard -fno-fast-math",
                         'cmp "$out/run-1.stdout" "$out/run-2.stdout"', "check_overfit.py"):
            self.assertIn(required, self.recipe)
        self.assertLess(self.recipe.index('600s "$out/overfit"'), self.recipe.index("check_overfit.py"))
        self.assertNotIn("byte_tokenizer_v1.tsv", self.recipe)
        self.assertNotIn("failure_bridge.c", self.recipe)
        self.assertNotIn("flock ", self.recipe)


if __name__ == "__main__":
    unittest.main()
