"""Cheap source/recipe checks; genuine Eshkol/native execution is a separate gate."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/cpu4_trainer_control"))
from test_source_contract import parse, walk

SOURCE = ROOT / "native/cpu4_training_step_extension.esk"
FIXTURE = ROOT / "tests/cpu4_step/runtime_test.esk"
RECIPE = ROOT / "scripts/test-cpu4-training-step.sh"


def fixture_check_budget(forms):
    """Count successful fixture assertions, including its fixed14 roster loop.

    This walks the actual fixture helper call graph. It does not execute model,
    native, failure, cursor or optimizer code and is not runtime admission.
    """
    definitions = {f[1][0]: f for f in forms if f[0] == "define" and isinstance(f[1], list)}

    def count(node, stack=()):
        if not isinstance(node, list) or not node:
            return 0
        head = node[0]
        if not isinstance(head, str):
            return sum(count(part, stack) for part in node)
        if head in ("define-syntax", "quote"):
            return 0
        if head == "define":
            return 0 if isinstance(node[1], list) else sum(count(part, stack) for part in node[2:])
        if head == "check":
            return 1
        if head == "let" and node[1] == "verify-roster-loop":
            # Exactly14 canonical slots in each of the two genuine setups.
            assert node[2] == [["index", "0"]]
            assert node[3][0:2] == ["if", ["<", "index", "14"]]
            return 14 * sum(count(part, stack) for part in node[3:])
        if head in definitions:
            if head in stack:
                return 0
            return sum(count(part, stack + (head,)) for part in definitions[head][2:])
        return sum(count(part, stack) for part in node[1:])

    return sum(count(form) for form in forms)


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

    def test_restore_decoder_uses_existing_core_error_boundary(self):
        binding = parse((ROOT / "native/cpu4_model_binding_extension.esk").read_text())
        restore = next(f for f in binding if f[0] == "define" and isinstance(f[1], list) and
                       f[1][0] == "cpu4-d2-cursor-restore-internal!")
        self.assertEqual(restore[2][0], "d2-with-core-errors")
        self.assertEqual(restore[2][1][:2], ["lambda", []])
        heads = [n[0] for n in walk(restore) if n and isinstance(n[0], str)]
        self.assertLess(heads.index("d2-core-cursor-decode"), heads.index("d2-token-dataset-seek!"))
        self.assertIn("cpu4-binding-fail", heads)
        self.assertIn("'cpu4-d2-cursor-restore!", (ROOT / "native/cpu4_model_binding_extension.esk").read_text())
        self.assertIn('"CPU4 identity error retains own operation"', self.fixture)
        self.assertIn("(transformer-error-operation caught)", self.fixture)

    def test_exact_successful_runtime_check_budget(self):
        previous = subprocess.run(["git", "show",
            "2574d0fb4eb94099802db18bf7a95fa3db096e63:tests/cpu4_step/runtime_test.esk"],
            cwd=ROOT, check=True, capture_output=True, text=True).stdout
        self.assertEqual(fixture_check_budget(parse(previous)), 118)
        self.assertEqual(fixture_check_budget(parse(self.fixture)), 140)
        self.assertIn("(not (= checks 140))", self.fixture)
        script = RECIPE.read_text()
        self.assertIn("grep -Fx 'CPU4 genuine full-step READ/WRITE/O2 PASS: checks=140'", script)
        self.assertNotIn("checks=[1-9]", script)


if __name__ == "__main__":
    unittest.main()
