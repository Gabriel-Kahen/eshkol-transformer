"""Causal source checks for the private held O2 to C2 claim boundary.

These checks do not replace a compiled same-owner transaction gate.
"""
from copy import deepcopy
from pathlib import Path
import unittest

from tests.c2.test_handler_order import definition, form, lists, parse


SOURCE = Path(__file__).resolve().parents[2] / "native/g3r_held_o2_adopter_extension.esk"
CLAIM = "g3r-held-o2-adopter-c2-claim!"
FAIL_STOP = form("(g3r-held-o2-fail-stop)")

OUTER = (
    "(not (eq? (vector-ref ledger 8) 'enrolled))",
    "(not (eq? (vector-ref request 0) c2-training-state-compose-tag))",
    "(not (eq? (vector-ref request 1) #f))",
    "(not (eq? (vector-ref request 2) #f))",
    "(not (eq? (car (vector-ref ledger 7)) (vector-ref ledger 6)))",
)
INNER = (
    "(not owner)",
    "(not (eq? (vector-ref owner 0) shell))",
    "(not (eq? (vector-ref owner 1) p1))",
    "(not (eq? (vector-ref owner 2) o2-shell))",
    "(not (eq? (vector-ref owner 7) 'live))",
    "(not (eq? (vector-ref (vector-ref owner 6) 0) c2-training-state-owner-token-tag))",
    "(not (eq? (vector-ref (vector-ref owner 6) 1) shell))",
    "(not (eq? (vector-ref (vector-ref owner 6) 2) #f))",
    "(not (eq? (vector-ref optimizer 1) (vector-ref ledger 3)))",
    "(not (eq? (vector-ref o2-state 0) o2-shell))",
    "(not (eq? o2-state (o2-find-identity o2-shell o2-states)))",
    "(not (eq? (vector-ref o2-state 1) issued))",
    "(not (eq? (vector-ref o2-state 6) (vector-ref owner 6)))",
    "(not issued)",
    "(not (= 0 (g3r-held-o2-issued-match (vector-ref ledger 1) (vector-ref ledger 3) issued (vector-ref o2-state 1))))",
)


def validate_claim(owner):
    if len(owner) != 4 or owner[2][0] != "if" or owner[2][2] != FAIL_STOP:
        raise ValueError("malformed fail-stop precondition")
    if owner[3][0] != "let*" or len(owner[3]) != 5:
        raise ValueError("malformed exact-owner lookup or transfer tail")
    body = owner[3]
    if body[2][0] != "if" or body[2][2] != FAIL_STOP:
        raise ValueError("postclaim mismatch does not fail stop")
    if body[3] != form("(vector-set! ledger 8 'claimed)") or body[4] != "shell":
        raise ValueError("claim transition is not the final assignment")
    outer, inner = owner[2][1], body[2][1]
    if outer[0] != "or" or inner[0] != "or":
        raise ValueError("claim guards must be fail-closed disjunctions")
    for text in OUTER:
        if outer.count(form(text)) != 1:
            raise ValueError(f"missing precondition: {text}")
    for text in INNER:
        if inner.count(form(text)) != 1:
            raise ValueError(f"missing owner check: {text}")
    return owner, outer, inner


def claim_guards(source):
    owner = definition({str(SOURCE): parse(source)}, CLAIM, str(SOURCE))
    return validate_claim(owner)


class HeldO2ClaimSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()

    def test_exact_claim_transition(self):
        claim_guards(self.source)
        release = definition({str(SOURCE): parse(self.source)},
                             "g3r-held-o2-adopter-release!", str(SOURCE))
        self.assertIn(form("'issued"), list(lists(release)))
        self.assertIn(form("'enrolled"), list(lists(release)))
        self.assertNotIn(form("'claimed"), list(lists(release)))

    def test_authority_mutations_are_rejected(self):
        original, _, _ = claim_guards(self.source)
        for location, checks in ((0, OUTER), (1, INNER)):
            for text in checks:
                with self.subTest(missing=text):
                    changed = deepcopy(original)
                    guard = changed[2][1] if location == 0 else changed[3][2][1]
                    guard.remove(form(text))
                    with self.assertRaises(ValueError):
                        validate_claim(changed)
        self.assertEqual(original[1], [CLAIM, "ledger", "request", "p1"])

    def test_no_commit_before_all_checks(self):
        original, _, _ = claim_guards(self.source)
        for location in ("outer", "inner", "tail"):
            with self.subTest(mutation=location):
                changed = deepcopy(original)
                if location == "outer":
                    changed[2][2] = "#f"
                elif location == "inner":
                    changed[3][2][2] = "#f"
                else:
                    changed[3][3] = "#t"
                with self.assertRaises(ValueError):
                    validate_claim(changed)


if __name__ == "__main__":
    unittest.main()
