"""Source-order contract for the uncompiled, test-only whole-call bracket."""

from copy import deepcopy
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tests/e3_private/fenv_driver.esk"


def definition(forms, name):
    return next(form for form in forms if isinstance(form, list)
                and form[:1] == ["define"] and isinstance(form[1], list)
                and form[1][0] == name)


def bracket_contract(form):
    region = form[2]
    assert region[0] == "with-region"
    token_binding = region[2]
    assert token_binding[0] == "let"
    assert token_binding[1] == [["token", ["e3-fenv-begin"]]]
    result_binding = token_binding[2]
    guard = result_binding[1][0][1]
    assert guard[0] == "guard"
    catch = guard[1][1]
    assert catch[1] == ["e3-fenv-end", "token", "1"]
    assert guard[2:4] == [
        ["e3-evaluate-into!", "model", "dataset", "frame"],
        ["e3-fenv-end", "token", "0"],
    ]
    assert result_binding[3][0] == "e3-fenv-report!"


class WholeCallSourceTest(unittest.TestCase):
    def test_guard_is_inside_region_and_end_precedes_catch_readback(self):
        forms = parse(DRIVER.read_text())
        invoke = definition(forms, "e3-fenv-invoke!")
        bracket_contract(invoke)
        self.assertEqual(DRIVER.read_text().count(
            "(e3-evaluate-into! model dataset frame)"), 2)
        self.assertNotIn("#ifdef", DRIVER.read_text())

    def test_order_mutations_are_rejected(self):
        invoke = definition(parse(DRIVER.read_text()), "e3-fenv-invoke!")
        moved = deepcopy(invoke)
        guard = moved[2][2][2][1][0][1]
        guard[2], guard[3] = guard[3], guard[2]
        with self.assertRaises(AssertionError):
            bracket_contract(moved)
        late_catch = deepcopy(invoke)
        catch = late_catch[2][2][2][1][0][1][1][1]
        catch[1], catch[2] = catch[2], catch[1]
        with self.assertRaises(AssertionError):
            bracket_contract(late_catch)


if __name__ == "__main__":
    unittest.main()
