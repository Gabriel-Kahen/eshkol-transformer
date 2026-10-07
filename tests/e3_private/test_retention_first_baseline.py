"""Source-only controls for the noninstalled retention driver's first call.

These checks do not replace a compiled mutation or supported runtime gate.
"""

from pathlib import Path
import re
import unittest


SOURCE = Path(__file__).with_name("retention_driver.esk")


def form(source: str, marker: str) -> str:
    start = source.index(marker)
    depth = 0
    string = comment = escape = False
    for end in range(start, len(source)):
        char = source[end]
        if comment:
            if char == "\n":
                comment = False
        elif string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                string = False
        elif char == ";":
            comment = True
        elif char == '"':
            string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return source[start:end + 1]
    raise ValueError("unclosed Eshkol form")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def audit_prepare(prepare: str) -> None:
    seed = prepare.index("(e3-test-seed-gradient")
    snapshot = prepare.index("(model-image (e3-test-retention-model-snapshot model))")
    first_call = prepare.index("(e3-test-retention-call!")
    calibration = prepare.index("(calibrated-region-bytes")
    probe_baseline = prepare.index("(before (e3-test-retention-probes))")
    require(seed < snapshot < first_call < calibration < probe_baseline,
            "model snapshot must precede first evaluation, probes follow calibration")
    require(prepare.index("'gradient-state") < first_call and
            "(if present? 1 0)" in prepare and
            prepare.index("'nonzero-gradient") < first_call and
            "1056964608" in prepare,
            "absent and seeded nonzero baseline controls precede evaluation")
    require("model-image rng before horizon present?" in prepare,
            "the pre-evaluation image must be retained for all later calls")


def audit_calls(source: str) -> None:
    call = form(source, "(define (e3-test-retention-call!")
    measured = call.index("(- (e3-test-region-used) region-before)")
    compared = call.index("'success-model-image")
    require(measured < compared < call.rindex("region-bytes"),
            "model preservation must follow the unchanged measured region")
    require("(e3-test-retention-model=? model model-image)" in call and
            "(with-region ('e3-retention-model-check 262144)" in call,
            "every completed call must compare the same model image")
    invocations = re.findall(r"\(e3-test-retention-call!", source)
    require(len(invocations) == 5, "unexpected retention evaluator call topology")
    for marker in ("(define (e3-test-retention-fail-retry!",
                   "(define (e3-test-retention-prepare",
                   "(define (e3-test-retention-execute"):
        body = form(source, marker)
        cursor = 0
        while "(e3-test-retention-call!" in body[cursor:]:
            cursor = body.index("(e3-test-retention-call!", cursor)
            expression = form(body[cursor:], "(e3-test-retention-call!")
            require(re.search(r"\bmodel-image\s*\)$", expression) is not None,
                    "all warmup, calibration, success and retry calls use baseline")
            cursor += len(expression)
    failure = form(source, "(define (e3-test-retention-fail-retry!")
    require("'failure-model-image" in failure and
            "(e3-test-retention-model=? model model-image)" in failure,
            "failed call must compare the original baseline")


def audit_image(source: str) -> None:
    snapshot = form(source, "(define (e3-test-retention-model-snapshot")
    compare = form(source, "(define (e3-test-retention-model=?")
    for body in (snapshot, compare):
        require("(= parameter 14)" in body and "(= gradient 2)" in body and
                "(= field 3)" in body and "(e3-test-gradient-metadata" in body,
                "all 14 parameter values, gradients and three metadata fields required")
        require("(e3-test-model-word-count" in body and
                re.search(r"\(e3-test-model-word\s+owner\s+parameter\s+"
                          r"gradient\s+index\)", body) is not None,
                "every value and gradient word required")


class FirstEvaluationBaseline(unittest.TestCase):
    def test_driver_forms_are_balanced(self) -> None:
        source = SOURCE.read_text()
        cursor = 0
        visited = 0
        while "(define " in source[cursor:]:
            cursor = source.index("(define ", cursor)
            cursor += len(form(source[cursor:], "(define "))
            visited += 1
        self.assertEqual(visited, source.count("(define "))

    def test_actual_driver_and_old_order_mutant(self) -> None:
        source = SOURCE.read_text()
        prepare = form(source, "(define (e3-test-retention-prepare")
        audit_prepare(prepare)
        audit_calls(source)
        audit_image(source)
        snapshot = "(model-image (e3-test-retention-model-snapshot model))"
        warmup = prepare.index("(e3-test-retention-call!")
        moved = prepare.replace(snapshot, "", 1)
        moved = moved[:warmup] + moved[warmup:].replace(
            "(let* ((calibrated-region-bytes", snapshot +
            "\n      (let* ((calibrated-region-bytes", 1)
        with self.assertRaises(ValueError):
            audit_prepare(moved)

    def test_missing_branch_or_postcall_checks_reject(self) -> None:
        source = SOURCE.read_text()
        prepare = form(source, "(define (e3-test-retention-prepare")
        for missing in ("'gradient-state", "'nonzero-gradient", "1056964608"):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                audit_prepare(prepare.replace(missing, "removed", 1))
        for missing in ("'success-model-image", "'failure-model-image"):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                audit_calls(source.replace(missing, "'removed", 1))
        for missing in ("(= parameter 14)", "(= gradient 2)", "(= field 3)",
                        "(e3-test-model-word owner parameter gradient index)"):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                if missing.startswith("(e3-test-model-word owner"):
                    mutant = re.sub(r"\(e3-test-model-word\s+owner\s+parameter\s+"
                                    r"gradient\s+index\)", "removed", source,
                                    count=1)
                else:
                    mutant = source.replace(missing, "removed", 1)
                audit_image(mutant)


if __name__ == "__main__":
    unittest.main()
