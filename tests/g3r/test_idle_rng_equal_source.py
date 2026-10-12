"""Source-only admission checks for the private G3-R no-draw predicate."""

from pathlib import Path
import unittest

from tests.c2.test_handler_order import definition, parse, render


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "native/g3r_idle_rng_equal_extension.esk"
EXPORTS = ROOT / "native/cli3_generate_public_exports.txt"


def verify(source: str) -> None:
    forms = parse(source)
    if len(forms) != 2 or forms[0] != [
        "extern", "i64", "g3r-native-idle-rng-equal", "ptr", "i64", "i64",
        "i64", "i64", ":real", "et_g3t_private_idle_rng_words_equal_v1",
    ]:
        raise ValueError("private status-only FFI signature differs")
    fn = definition({"extension": forms}, "g3r-idle-rng-words-equal-internal!")
    if fn[1] != ["g3r-idle-rng-words-equal-internal!", "generator", "version",
                 "seed", "counter-low", "counter-high"]:
        raise ValueError("private five-input entry differs")
    body = render(fn)
    required = (
        "(m3-call operation",
        "(m3t-exact-i64? version)", "(= version 1)",
        "(m3t-exact-i64? seed)", "(>= seed 0)",
        "(m3t-exact-i64? counter-low)",
        "(m3t-exact-i64? counter-high)",
        "(g3t-prefill-entry generator (quote generator) operation)",
        "(eq? (vector-ref entry 2) (quote live))",
        "(not (null? (vector-ref entry 3)))",
        "(eq? (vector-ref entry 4) #f)",
        "(eq? (vector-ref entry 8) #f)",
        "(eq? (vector-ref entry 9) #f)",
        "(eq? (vector-ref entry 10) #f)",
        "(eq? (vector-ref entry 11) (quote diagnostic-c2))",
        "(g3t-model-entry-live (vector-ref model-entry 0) operation #f)",
        "(t1-private-entry tokenizer)",
        "(g3r-native-idle-rng-equal (vector-ref entry 3) version seed counter-low counter-high)",
        "(g3t-native-fail-raw operation)",
    )
    if any(item not in body for item in required):
        raise ValueError("exact generator/word admission or native error mapping is missing")
    if not (body.index("(m3t-exact-i64? version)")
            < body.index("(g3t-prefill-entry generator")
            < body.index("(g3t-model-entry-live")
            < body.index("(t1-private-entry tokenizer)")
            < body.index("(g3r-native-idle-rng-equal")
            < body.index("(g3t-native-fail-raw operation)")):
        raise ValueError("admission must precede native comparison")
    if any(item in body for item in ("vector-set!", "set!", "generator-generate!",
                                     "g3t-test-", "g3r-record-decode!")):
        raise ValueError("idle equality entry cannot mutate or decode authority")


class IdleRngEqualSourceTest(unittest.TestCase):
    def test_private_admission_and_no_public_export(self) -> None:
        verify(EXTENSION.read_text())
        self.assertNotIn("g3r", EXPORTS.read_text())

    def test_missing_or_reordered_authority_is_rejected(self) -> None:
        source = EXTENSION.read_text()
        for original, replacement in (
            ("(g3t-prefill-entry generator 'generator operation)",
             "(g3t-generator-find generator operation)"),
            ("(eq? (vector-ref entry 9) #f)", "#t"),
            ("(g3t-model-entry-live\n                             (vector-ref model-entry 0) operation #f)",
             "model-entry"),
            ("(t1-private-entry tokenizer)", "#t"),
            ("(g3t-native-fail-raw operation)", "#t"),
        ):
            self.assertIn(original, source)
            with self.subTest(original=original):
                with self.assertRaises(ValueError):
                    verify(source.replace(original, replacement, 1))


if __name__ == "__main__":
    unittest.main()
