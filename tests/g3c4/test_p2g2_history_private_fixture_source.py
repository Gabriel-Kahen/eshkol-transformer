"""Source controls for the uncompiled, one-root private history fixture."""

from importlib.machinery import SourceFileLoader
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/g3c4/p2g2_public_history_private_runtime.esk"
parse = SourceFileLoader(
    "p2g2_history_fixture_forms",
    str(ROOT / "scripts/check-g3c4-p2g2-terminal-close.py")
).load_module()


def source() -> str:
    return FIXTURE.read_text()


def walk(node):
    if isinstance(node, list):
        yield node
        for child in node:
            yield from walk(child)


def validate(text: str) -> None:
    forms = parse.forms(text)
    loads = [node for form in forms for node in walk(form)
             if node and node[0] == "load"]
    assert loads == [["load", '"g3g_manual_package_root.esk"']]
    assert not any(node and node[0] == "require"
                   for form in forms for node in walk(form))
    assert "(define original-t1" in text
    assert "(define foreign-input" in text
    assert "(t1-wave1-tokenizer-encode tokenizer (bytevector 0 255))" in text
    assert "(g3c4-public-generation-input-create original-t1)" in text
    assert "(g3c4-with-p2g2-prefix-call 'private-history-first-reference" in text
    assert "(g3g-public-generator-generate generator input)" in text
    assert "(g3c4-public-p2g2-history-capture-internal!" in text
    assert "(g3c4-public-p2g2-history-admit-once-internal!" in text
    assert "(run-case selected-first 1 'false)" in text
    assert "(run-case non-eos 2 'late)" in text
    assert "(if (= selected-first 255) 0 255)" in text
    for label in ("copied history is not issued",
                  "capture final snapshot mutation denies issuance",
                  "capture used initial and final snapshot exactly once",
                  "foreign generator cannot consume genuine history",
                  "wrong live input cannot capture genuine output",
                  "wrong live output cannot capture genuine origin",
                  "wrong live input cannot consume genuine history",
                  "wrong live output cannot consume genuine history",
                  "private catch retains original raised #f",
                  "last provider mutation denies before spend",
                  "reentrant admit denied while outer owns phase",
                  "one-use history cannot admit twice"):
        assert f'"{label}"' in text, label
    mutation = text.find("(vector-set! ticket 14")
    denial = text.find('"last provider mutation denies before spend"')
    repair = text.find("(vector-set! ticket 14 (vector-ref before 0))")
    reentry = text.find('"reentrant admit denied while outer owns phase"')
    assert 0 <= mutation < denial < repair < reentry
    capture_mutation = text.find("(vector-set! ticket 15")
    capture_denial = text.find(
        '"capture final snapshot mutation denies issuance"')
    capture_repair = text.find(
        "(vector-set! ticket 15 (vector-ref before 1))")
    authentic_capture = text.find("(set! history")
    assert 0 <= capture_mutation < capture_denial < capture_repair < authentic_capture
    capture = text[:authentic_capture]
    assert "(set! snapshot-calls (+ snapshot-calls 1))" in capture
    assert "(if (= snapshot-calls 2)" in capture
    assert "(= snapshot-calls 2))" in capture
    assert text.index("(if (= snapshot-calls 2)") < capture_mutation
    assert text.index('"private catch retains original raised #f"') < (
        text.index("(set! g3c4-p2g2-terminal-snapshot-private snapshot-provider)",
                   text.index('"private catch retains original raised #f"')))
    assert text.count("(set! g3c4-p2g2-terminal-snapshot-private snapshot-provider)") >= 3
    assert "(g3g-public-generation-tensor-release original-t1)" not in text
    assert "(t1-shell-abort original-t1)" not in text
    assert text.index("(check \"descriptor preserves full P2/G1-G2 history\"") < (
        text.index("(g3g-public-generation-output-release output)"))
    assert text.index('"wrong live input cannot capture genuine output"') < (
        text.index("(set! history"))
    assert text.index('"wrong live output cannot capture genuine origin"') < (
        text.index("(set! history"))
    assert text.index('"wrong live input cannot consume genuine history"') < (
        text.index('(check "descriptor preserves full P2/G1-G2 history"'))
    assert text.index('"wrong live output cannot consume genuine history"') < (
        text.index('(check "descriptor preserves full P2/G1-G2 history"'))
    for label, expected in (
        ("wrong live input cannot capture genuine output",
         "generator foreign-input output"),
        ("wrong live output cannot capture genuine origin",
         "generator input foreign-output"),
        ("wrong live input cannot consume genuine history",
         "history generator foreign-input output"),
        ("wrong live output cannot consume genuine history",
         "history generator input foreign-output"),
    ):
        start = text.index(f'"{label}"')
        assert expected in text[start:start + 260], label
    assert "(g3g-public-generation-output-release foreign-output)" in text
    assert "(g3g-public-generation-tensor-release foreign-input)" in text
    nested = text[text.index("(define (run-nested-capture)"):]
    assert "(run-nested-capture)" in text
    assert "(if (= snapshot-calls 2)" in nested
    assert "(set! inner-history" in nested
    for label in ("outer capture denies after nested sole issuance",
                  "outer capture reached final snapshot exactly once",
                  "nested capture retained its issued shell",
                  "same output cannot issue a second history",
                  "nested sole issue retains exact output and prompt",
                  "nested capture did not change terminal state",
                  "nested issued history spends once"):
        assert f'"{label}"' in nested
    assert nested.index("(set! inner-history") < nested.index(
        '"outer capture denies after nested sole issuance"')
    assert nested.index(
        "(set! g3c4-p2g2-terminal-snapshot-private snapshot-provider)"
    ) < nested.index("(set! inner-history")
    assert nested.index('"same output cannot issue a second history"') < (
        nested.index("(let* ((descriptor"))
    for index, expected in enumerate((
            "0", "255", "(vector-ref before 0)", "(vector-ref before 1)",
            "expected-count", "(+ expected-count 2)",
            "(vector-ref before 4)", "(vector-ref before 5)",
            "(vector-ref before 6)", "(vector-ref before 7)", "eos")):
        assert f"(= (vector-ref descriptor {index}) {expected})" in text
    for index, owner in ((11, "model"), (12, "tokenizer"),
                         (13, "generator"), (14, "input"), (15, "output")):
        assert f"(eq? (vector-ref descriptor {index}) {owner})" in text
    assert "(if (= index 8) #t" in text
    assert "(= (vector-ref before index) (vector-ref after index))" in text


class FixtureSource(unittest.TestCase):
    def test_actual_fixture(self):
        validate(source())

    def test_branch_and_one_use_omissions(self):
        text = source()
        for needle in ('(run-case selected-first 1 \'false)',
                       '(run-case non-eos 2 \'late)',
                       '"one-use history cannot admit twice"'):
            with self.assertRaises(AssertionError):
                validate(text.replace(needle, "omitted", 1))

    def test_last_provider_restore_and_reentry_omissions(self):
        text = source()
        for needle in ('"last provider mutation denies before spend"',
                       '"capture final snapshot mutation denies issuance"',
                       '(vector-set! ticket 14 (vector-ref before 0))',
                       '(vector-set! ticket 15 (vector-ref before 1))',
                       '"reentrant admit denied while outer owns phase"'):
            with self.assertRaises(AssertionError):
                validate(text.replace(needle, "omitted", 1))

    def test_descriptor_slot_map_mutants(self):
        text = source()
        for old, new in (
            ("(vector-ref descriptor 2) (vector-ref before 0)",
             "(vector-ref descriptor 2) (vector-ref before 3)"),
            ("(vector-ref descriptor 3) (vector-ref before 1)",
             "(vector-ref descriptor 3) (vector-ref before 2)"),
            ("(vector-ref descriptor 5) (+ expected-count 2)",
             "(vector-ref descriptor 5) expected-count"),
            ("(vector-ref descriptor 9) (vector-ref before 7)",
             "(vector-ref descriptor 9) (vector-ref before 6)"),
        ):
            with self.assertRaises(AssertionError):
                validate(text.replace(old, new, 1))
        for old, new in (
            ("(eq? (vector-ref descriptor 11) model)",
             "(eq? (vector-ref descriptor 11) tokenizer)"),
            ("(eq? (vector-ref descriptor 12) tokenizer)",
             "(eq? (vector-ref descriptor 12) model)"),
        ):
            with self.assertRaises(AssertionError):
                validate(text.replace(old, new, 1))
            with self.assertRaises(AssertionError):
                validate(text.replace(old, "#t", 1))

    def test_final_only_capture_fault_control(self):
        text = source()
        with self.assertRaises(AssertionError):
            validate(text.replace("(if (= snapshot-calls 2)",
                                  "(if (= snapshot-calls 1)", 1))
        with self.assertRaises(AssertionError):
            validate(text.replace("(= snapshot-calls 2))",
                                  "(= snapshot-calls 1))", 1))

    def test_root_and_t1_lifecycle_controls(self):
        text = source()
        with self.assertRaises(AssertionError):
            validate(text.replace('(load "g3g_manual_package_root.esk")',
                                  '(load "g3g_manual_package_root.esk")\n'
                                  '(load "g3g_manual_package_root.esk")', 1))
        with self.assertRaises(AssertionError):
            validate(text + "\n(t1-shell-abort original-t1)\n")

    def test_wrong_owner_and_nested_capture_mutants(self):
        text = source()
        for needle in (
            '"wrong live input cannot capture genuine output"',
            '"wrong live output cannot capture genuine origin"',
            '"wrong live input cannot consume genuine history"',
            '"wrong live output cannot consume genuine history"',
            '"same output cannot issue a second history"',
            '(set! inner-history',
            '"nested issued history spends once"',
        ):
            with self.assertRaises(AssertionError):
                validate(text.replace(needle, "omitted", 1))
        for label, old, new in (
            ("wrong live input cannot capture genuine output",
             "generator foreign-input output", "generator input output"),
            ("wrong live output cannot capture genuine origin",
             "generator input foreign-output", "generator input output"),
            ("wrong live input cannot consume genuine history",
             "history generator foreign-input output",
             "history generator input output"),
            ("wrong live output cannot consume genuine history",
             "history generator input foreign-output",
             "history generator input output"),
        ):
            start = text.index(f'"{label}"')
            at = text.index(old, start)
            with self.assertRaises(AssertionError):
                validate(text[:at] + new + text[at + len(old):])
        nested = text.index("(define (run-nested-capture)")
        with self.assertRaises(AssertionError):
            validate(text[:nested] + text[nested:].replace(
                "(if (= snapshot-calls 2)", "(if (= snapshot-calls 1)", 1))


if __name__ == "__main__":
    unittest.main()
