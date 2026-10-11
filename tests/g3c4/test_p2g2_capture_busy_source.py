"""Parsed source controls for the private capture reservation (no compiler claim)."""

from importlib.machinery import SourceFileLoader
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/g3c4_public_g2_result_extension.esk"
parse = SourceFileLoader(
    "capture_busy_forms", str(ROOT / "scripts/check-g3c4-p2g2-terminal-close.py")
).load_module()


def route(text):
    return next(form for form in parse.forms(text)
                if form[0] == "define"
                and form[1] == "g3c4-public-p2g2-history-route")


def validate(text):
    top = route(text)
    closure = top[2]
    assert ["capture-busy", "#f"] in closure[1]
    action = closure[-1][-1]
    assert action[0] == "let"
    clauses = action[-1]
    assert clauses[0] == "cond"
    assert [arm[0] for arm in clauses[1:]] == [
        ["eq?", "action", "'input-issue"],
        ["eq?", "action", "'capture"],
        ["eq?", "action", "'admit"], "else"]
    capture = clauses[2]
    assert len(capture) == 4
    assert capture[1][0:2] == ["if", "capture-busy"]
    assert capture[2][0] == "if"
    assert [">=", "history-count", "64"] in capture[2][1][1:]
    assert ["memq", "third", "issued-outputs"] in capture[2][1][1:]
    owned = capture[3]
    assert owned[0] == "let"
    assert ["capture-phase", "'ready"] in owned[1]
    assert ["token", ["vector", "'g3c4-p2g2-capture-busy"]] in owned[1]
    guard = owned[2]
    assert guard[0] == "guard"
    assert guard[2] == ["set!", "capture-phase", "'reserved"]
    assert guard[3] == ["set!", "capture-busy", "token"]
    handler = guard[1]
    assert handler[0] == "caught"
    assert handler[1][1] == ["if", ["not", ["eq?", "capture-phase", "'reserved"]],
                             ["exit", "134"]]
    assert ["raise", "caught"] == handler[1][-1]
    assert handler[1][2][1][1] == ["set!", "capture-busy", "#f"]
    assert handler[1][2][-1] == ["else", ["exit", "134"]]
    providers = guard[4]
    assert providers[0] == "let*"
    names = [binding[0] for binding in providers[1]]
    assert names.index("input-entry") < names.index("snapshot")
    assert ["snapshot", ["g3c4-public-g2-source-snapshot", "output", "operation"]] in providers[1]
    shell = providers[3]
    assert shell[0] == "let*" and len(shell) == 3
    assert ["phase", "'live"] in shell[1]
    assert ["final-snapshot", ["g3c4-p2g2-terminal-snapshot-private", "output"]] in shell[1]
    tail = shell[2]
    assert tail[0] == "let"
    assert ["next-count", ["+", "history-count", "1"]] in tail[1]
    checks = tail[2]
    assert checks[0] == "if"
    required = checks[1][1]
    assert ["eq?", "capture-busy", "token"] in required[1:]
    assert ["eq?", "capture-phase", "'reserved"] in required[1:]
    assert ["<", "history-count", "64"] in required[1:]
    assert tail[3:] == [
        ["set!", "capture-phase", "'committing"],
        ["set!", "history-issuers", "next-history"],
        ["set!", "issued-outputs", "next-outputs"],
        ["set!", "history-count", "next-count"],
        ["set!", "capture-busy", "#f"],
        ["set!", "capture-phase", "'done"], "shell"]
    # After the final predicate there is no provider call or allocation.
    return capture


class CaptureBusySource(unittest.TestCase):
    def test_actual_route(self):
        validate(SOURCE.read_text())

    def test_preprovider_and_tail_mutants(self):
        text = SOURCE.read_text()
        for before, after in (
            ("(if capture-busy", "(if #f"),
            ("(>= history-count 64)", "(>= history-count 65)"),
            ("(set! capture-busy token)", "(set! capture-busy #f)"),
            ("(set! capture-phase 'committing)",
             "(set! capture-phase 'reserved)"),
            ("(if (not (eq? capture-phase 'reserved)) (exit 134))",
             "(if (not (eq? capture-phase 'reserved)) #t)"),
            ("(set! capture-busy #f)\n                 (set! capture-phase 'done)",
             "(set! capture-busy token)\n                 (set! capture-phase 'done)"),
            ("(eq? capture-busy token)\n                       (eq? capture-phase 'reserved)",
             "(eq? capture-busy #f)\n                       (eq? capture-phase 'reserved)"),
        ):
            self.assertIn(before, text)
            with self.subTest(before=before), self.assertRaises(AssertionError):
                validate(text.replace(before, after, 1))

    def test_original_false_and_sibling_arm_controls(self):
        text = SOURCE.read_text()
        for before, after in (
            ("(raise caught)", "(raise 'replaced)"),
            ("((eq? action 'admit)", "((eq? action 'capture)"),
            ("(capture-phase 'ready)", "(capture-phase 'reserved)"),
        ):
            self.assertIn(before, text)
            at = text.index(before, text.index("((eq? action 'capture)"))
            with self.subTest(before=before), self.assertRaises(AssertionError):
                validate(text[:at] + after + text[at + len(before):])


if __name__ == "__main__":
    unittest.main()
