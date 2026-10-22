"""Cheap source/recipe checks; genuine Eshkol/native execution is a separate gate."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/cpu4_trainer_control"))
from test_source_contract import parse, walk

SOURCE = ROOT / "native/cpu4_training_step_extension.esk"
FIXTURE = ROOT / "tests/cpu4_step/runtime_test.esk"
RECIPE = ROOT / "scripts/test-cpu4-training-step.sh"


class StepSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text()
        cls.forms = parse(cls.text)
        cls.fixture = FIXTURE.read_text()
        cls.recipe = RECIPE.read_text()

    def test_all_ifs_have_valid_arity_and_definitions_are_unique(self):
        names = []
        for form in self.forms + parse(self.fixture):
            for node in walk(form):
                if node and node[0] == "if":
                    self.assertIn(len(node), (3, 4), node)
        for form in self.forms:
            if form[0] in ("define", "define-syntax"):
                names.append(form[1][0] if isinstance(form[1], list) else form[1])
        self.assertEqual(len(names), len(set(names)))

    def test_no_old_authority_or_callback_and_real_dependencies(self):
        self.assertNotIn("(extern ", self.text)
        for forbidden in ("tr3-lease-", "m3t-entry", "m3t-public-model", "test-hook"):
            self.assertNotIn(forbidden, self.text)
        heads = {n[0] for f in self.forms for n in walk(f) if n and isinstance(n[0], str)}
        for required in ("cpu4-trainer-idle-check-internal", "cpu4-model-binding-check-internal",
                         "cpu4-training-microbatch-internal!", "cpu4-control-native-enter",
                         "cpu4-control-native-leave", "o2-trainer-step-clear-prepare-internal",
                         "o2-trainer-step-clear-commit-internal!", "o2-trainer-step-clear-abort-internal!",
                         "cpu4-control-validate-copy", "cpu4-d2-cursor-restore-internal!",
                         "o2-public-optimizer-zero-grad!"):
            self.assertIn(required, heads)
        top = next(f for f in self.forms if f[0] == "define" and
                   isinstance(f[1], list) and f[1][0] == "cpu4-training-step-internal!")
        self.assertEqual(top[1], ["cpu4-training-step-internal!", "trainer"])

    def test_actual_observations_are_retained_and_no_float_reduction(self):
        self.assertIn("(observations (make-vector accumulation #f))", self.text)
        self.assertIn("(result (vector 0 0 0 observations))", self.text)
        self.assertIn("(make-bytevector 12 0)", self.text)
        self.assertNotIn("bytevector-ieee", self.text)
        self.assertNotIn("numerator", self.text)

    def test_no_recoverable_postcommit_tail(self):
        commit = next(f for f in self.forms if f[0] == "define" and
                      isinstance(f[1], list) and f[1][0] == "cpu4-training-step-commit-internal!")
        # fe9 handler installation may malloc: it must precede native commit.
        # The already-bound flag selects categorized pre-write rethrow versus
        # fail-stop for any impossible error after a completed native update.
        body = commit[2][1][2:]
        bound = body[-1]
        self.assertEqual(bound[:2], ["let", [["committed?", "#f"]]])
        guarded = bound[2]
        self.assertEqual(guarded[0], "guard")
        handler = guarded[1]
        self.assertEqual(handler[1][1],
                         ["if", "committed?", ["cpu4-binding-fail-stop", "70"], ["raise", "caught"]])
        self.assertEqual(guarded[2][0], "o2-trainer-step-clear-commit-internal!")
        self.assertEqual(guarded[3], ["set!", "committed?", "#t"])
        heads = {n[0] for f in guarded[3:] for n in walk(f) if n and isinstance(n[0], str)}
        for forbidden in ("make-vector", "vector", "cons", "make-bytevector",
                          "cpu4-control-validate-copy", "cpu4-d2-cursor-capture-internal",
                          "o2-public-optimizer-zero-grad!", "raise", "guard"):
            self.assertNotIn(forbidden, heads)
        self.assertIn("(> accumulation (quotient cpu4-step-exact-weight-max 4))", self.text)
        self.assertEqual(4 * 4194304, 16777216)
        self.assertGreater(4 * 4194305, 16777216)

    def test_fixture_uses_real_bytewise_full_state_and_failure_operations(self):
        self.assertIn("(t2-wave2-tokenizer-byte resolved)", self.fixture)
        self.assertNotIn("t2-wave2-tokenizer-load", self.fixture)
        self.assertIn("(d2-core-bytes=? (vector-ref left index) (vector-ref right index))", self.fixture)
        self.assertNotIn("(equal? before", self.fixture)
        for label in ("failed D2 seek retains frame", "zero-grad allocation failure",
                      "real O2 prepare allocation failure", "rejected plan abort retains WRITE",
                      "full-step prewrite failure rolls back", "failure after EOS rewind rolls back",
                      "all1192 parameter bytes equal independent O2 path",
                      "all2384 moment words and optimizer updates equal"):
            self.assertIn('"' + label + '"', self.fixture)
        self.assertIn("(o2-public-optimizer-step! (vector-ref reference 4))", self.fixture)

    def test_recipe_native_identity_and_failpoints_are_explicit(self):
        for required in ("native/cpu4_training_native_sources.txt", "-DET_CPU4_OBJECTIVE_PRIVATE",
                         "-DET_TR3_O2_STEP_CLEAR_NATIVE -DET_O2_TESTING",
                         "-DET_O2_NATIVE_HELPERS_ONLY -DET_TR3_O2_STEP_CLEAR_BRIDGE",
                         "tests/cpu4_step/failure_bridge.c", "--strict-types --optimize 0 --no-stdlib",
                         "-ffp-contract=off -fexcess-precision=standard -fno-fast-math",
                         'cmp "$out/run-1.stdout" "$out/run-2.stdout"'):
            self.assertIn(required, self.recipe)
        self.assertNotIn("byte_tokenizer_v1.tsv", self.recipe)
        self.assertNotIn("flock ", self.recipe)

    def test_finite_cursor_equation_reference(self):
        # Independent integer reference for boundary and mid-step EOS cases.
        for span in (1, 2, 3, 17):
            for accumulation in (1, 2, 3, 7):
                position = epochs = 0
                for update in range(1, 31):
                    for _ in range(accumulation):
                        if position == span:
                            position = 0
                            epochs += 1
                        position += 1
                    self.assertEqual(update * accumulation, epochs * span + position)


if __name__ == "__main__":
    unittest.main()
