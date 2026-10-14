"""Source controls for the noninstalled post-retention teardown phase."""

from pathlib import Path
import unittest

from tests.e3_private.test_retention_first_baseline import form


HERE = Path(__file__).parent
DRIVER = HERE / "retention_driver.esk"
FIXTURE = HERE / "driver.esk"
RUNTIME = HERE / "retention_runtime.esk"


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def audit(driver: str, fixture: str, runtime: str) -> None:
    report = form(driver, "(define (e3-test-retention-report")
    teardown = form(driver, "(define (e3-test-retention-teardown")
    dispatch = form(driver, "(define (e3-test-retention-run")
    destroy = form(fixture, "(define (e3-test-destinations-destroy!")
    require("(vector-set! record 17 (e3-test-retention-probes))" in report and
            "(vector-set! record 18 #t)" in report and
            "(not (vector-ref record 18))" in report and
            "(vector-set! e3-test-retention-root 0 #f)" not in report,
            "report must retain the post-close 29-probe baseline")
    require("(vector-ref record 17)" in teardown and
            "(vector-ref record 18)" in teardown and
            "(eq? phase 'teardown)" in dispatch and
            "(vector 'retention 'teardown horizon present?)" in runtime,
            "teardown must run after report against the retained record")
    unbind = teardown.find("(e3-frame-unbind! frame)")
    destinations = teardown.find("(e3-test-destinations-destroy! destinations)")
    after = teardown.find("(after (e3-test-retention-probes))")
    require(0 <= unbind < destinations < after,
            "both owned resources must be retired before the final probe")
    require("(= (- (vector-ref before 0) (vector-ref after 0)) 41)" in teardown and
            "(= (vector-ref after 18) (+ (vector-ref before 18) 1))" in teardown and
            "(= (vector-ref after 17) (- (vector-ref before 17) 1))" in teardown,
            "all 37 frame tensors, four destinations and one frame must retire")
    for index in (1, 2):
        require(f"(= (vector-ref before {index}) (vector-ref after {index}))" in teardown,
                "model-owned counts must remain unchanged")
    require("(e3-test-retention-model=? model (vector-ref record 6))" in teardown,
            "model words and gradient metadata must remain unchanged")
    for index in (3, 21, 22, 23, 24, 25, 26):
        require(f"(= (vector-ref after {index}) 0)" in teardown,
                "borrow, graph, dataset, batch and FD authorities must be zero")
    require("(= index 4)" in destroy and
            "(e3-test-destination-destroy" in destroy and
            "(vector-ref destinations index)" in destroy and
            "(cleanup (+ index 1))" in destroy,
            "the accepted destructor must visit and destroy all four destinations")
    require(runtime.index("(e3-retention-test-run report input output)") <
            runtime.index("(e3-retention-test-run teardown input output)") <
            runtime.index('"E3-COMPLETE-CALL-RETENTION-PASS horizon="'),
            "runtime success must follow the final teardown")


class RetentionTeardownSourceTest(unittest.TestCase):
    def test_contract_and_causal_mutants(self) -> None:
        driver, fixture, runtime = (path.read_text() for path in
                                    (DRIVER, FIXTURE, RUNTIME))
        audit(driver, fixture, runtime)
        teardown = "(e3-frame-unbind! frame)"
        with self.assertRaisesRegex(ValueError, "both owned resources"):
            audit(driver.replace(teardown, "#t", 1), fixture, runtime)
        with self.assertRaisesRegex(ValueError, "accepted destructor"):
            audit(driver, fixture.replace("(= index 4)", "(= index 3)", 1), runtime)
        with self.assertRaisesRegex(ValueError, "both owned resources"):
            audit(driver.replace("(e3-test-destinations-destroy! destinations)",
                                 "#t", 1), fixture, runtime)


if __name__ == "__main__":
    unittest.main()
