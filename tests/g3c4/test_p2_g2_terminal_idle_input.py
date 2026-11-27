"""Source contracts for the idle-only P2/G2 gate."""
from pathlib import Path
import subprocess
import unittest

from tests.c2.test_handler_order import definition, form, lists, parse

ROOT = Path(__file__).resolve().parents[2]
LEAF = "native/g3c4_p2g2_terminal_idle_input_extension.esk"
FIXTURE = "tests/g3c4/p2_g2_terminal_idle_input_test.esk"
RUNNER = "scripts/test-g3c4-p2g2-t1-prefix-coordinator.sh"
PREFLIGHT = "g3c4-p2g2-terminal-idle-input-preflight"


class TerminalIdleInputTests(unittest.TestCase):
    def setUp(self):
        self.text = (ROOT / LEAF).read_text()
        self.body = definition({LEAF: parse(self.text)}, PREFLIGHT)
        self.forms = list(lists(self.body[2:]))

    def test_read_only_native_preflight_brackets_the_lexical_recheck(self):
        self.assertEqual(self.body[1], [PREFLIGHT, "generator", "input", "operation"])
        self.assertEqual(self.body[2][1], form("(not (g3c4-call-tuple? 2 2))"))
        locals_body = self.body[3]
        self.assertEqual(locals_body[2], form("""
            (define (native-idle!)
              (g3c4-native-check
                (g3c4-p2g2-native-prefill-preflight generator-native input-native 2)
                operation))
        """))
        self.assertEqual(locals_body[4], ["native-idle!"])
        recheck = locals_body[5][2]
        self.assertEqual(recheck[1], ["current-generator", "current-input"])
        self.assertEqual(recheck[3], ["native-idle!"])
        self.assertEqual(recheck[4], form("""
            (vector generator generator-entry generator-native
                    model model-entry model-native tokenizer tokenizer-entry core
                    policy input input-entry input-native)
        """))
        self.assertNotRegex(self.text, r"\((?:extern|set!|vector-set!|m3-call|provide|export)\s")
        self.assertNotRegex(self.text, r"g3c4-\S*(?:acquire|reserve|commit|retire)!?\s")
        native = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
        hook = native.split("int64_t et_g3c4_private_prompt_prefill_preflight_v1(", 1)[1]
        hook = hook.split("\n}\n", 1)[0]
        for predicate in ("et_g3c4_admit_idle_call(context_candidate)",
                          "et_g3c4_admit_input(input_candidate, 0)",
                          "ET_G3C4_P2_G2_TUPLE(input->length, budget)",
                          "budget == 2 && context->generator_policy[4] != 2"):
            self.assertIn(predicate, hook)
        self.assertNotRegex(hook, r"(?:acquire|borrow|dispatch|commit|reserve)\(")

    def test_original_identities_and_policy_values_are_checked(self):
        for predicate in (
            "(eq? current-generator generator)",
            "(eq? current-input input)",
            "(eq? (g3c4-generator-entry-call-ready current-generator operation) generator-entry)",
            "(eq? (vector-ref generator-entry 3) generator-native)",
            "(eq? (vector-ref generator-entry 5) model-entry)",
            "(eq? (vector-ref model-entry 0) model)",
            "(eq? (vector-ref model-entry 3) model-native)",
            "(eq? (vector-ref generator-entry 6) tokenizer)",
            "(eq? (g3c4-baseline-tokenizer-entry tokenizer operation 'invalid-state 'invalid-state)"
            " tokenizer-entry)",
            "(eq? (car (cdr tokenizer-entry)) core)",
            "(eq? (vector-ref generator-entry 7) policy)",
            "(equal? policy policy-values)",
            "(eq? (g3c4-input-entry-live current-input operation) input-entry)",
            "(eq? (vector-ref input-entry 3) input-native)",
        ):
            self.assertIn(form(predicate), self.forms)
        self.assertIn(form("(policy-values (vector-copy policy))"), self.forms)
        self.assertIn(form("(input-entry (g3c4-input-entry-live input operation))"), self.forms)
        witness = (ROOT / FIXTURE).read_text()
        self.assertGreaterEqual(witness.count("(g3c4-p2g2-owned-input-create"), 3)
        self.assertNotRegex(witness, r"\(g3c4-\S*(?:acquire|reserve|commit|retire)!?\s")

    def test_only_the_private_fixture_loads_the_leaf(self):
        result = subprocess.check_output(
            ["git", "grep", "--untracked", "-IlzE", "g3c4[-_]p2g2[-_]terminal[-_]idle[-_]input",
             "--", "."], cwd=ROOT, text=True,
        )
        self.assertEqual(set(result.rstrip("\0").split("\0")), {
            LEAF, FIXTURE, RUNNER, Path(__file__).relative_to(ROOT).as_posix(),
        })
        self.assertEqual((ROOT / FIXTURE).read_text().count(f'(load "{Path(LEAF).name}")'), 1)


if __name__ == "__main__":
    unittest.main()
