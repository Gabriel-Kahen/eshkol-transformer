"""Source-only checks for the unadmitted E3 fault driver successor."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/e3_private"


class FaultDriverSourceTest(unittest.TestCase):
    def test_exact_root_and_bridge_layer(self) -> None:
        self.assertEqual(
            (SOURCE / "fault_driver_root.esk").read_text().splitlines()[1:],
            ['(load "retention_driver_root.esk")',
             '(load "fault_driver.esk")'],
        )
        bridge = (SOURCE / "fault_bridge.c").read_text()
        self.assertEqual(bridge.count('#include "retention_bridge.c"'), 1)
        self.assertIn("et_e3_private_fault_dispatch_cabi_v1", bridge)
        self.assertIn("void et_e3_fault_run_v1(", bridge)

    def test_scope_closes_before_status_or_retry(self) -> None:
        source = (SOURCE / "fault_driver.esk").read_text()
        start = source.index("(define (e3-test-fault-scoped-call ")
        end = source.index("(define (e3-test-fault-status=?", start)
        scope = source[start:end]
        self.assertLess(scope.index("(e3-fault-arm native"),
                        scope.index("(e3-fault-begin native"))
        self.assertLess(scope.index("(e3-fault-begin native"),
                        scope.index("(guard"))
        self.assertLess(scope.index("(guard"),
                        scope.index("(e3-evaluate-into!"))
        self.assertLess(scope.index("(e3-evaluate-into!"),
                        scope.index("(e3-fault-end native"))
        guard_start = scope.index("(guard")
        depth = 0
        guard_end = None
        for index in range(guard_start, len(scope)):
            if scope[index] == "(":
                depth += 1
            elif scope[index] == ")":
                depth -= 1
                if depth == 0:
                    guard_end = index + 1
                    break
        self.assertIsNotNone(guard_end)
        self.assertLess(guard_end, scope.index("(e3-fault-end native"))
        self.assertNotIn("e3-fault-status", scope)
        self.assertNotIn("e3-fault-reset", scope)
        self.assertIn("(vector-ref (e3-frame-entry-live frame", source)
        self.assertIn("(e3-test-fault-scoped-call model dataset frame 20 8192)",
                      source)
        self.assertIn("'selected-stage-and-native-status", source)
        self.assertIn("'same-frame-retry", source)
        output = source[source.index("(define (e3-test-fault-output "):
                        source.index("(define (e3-test-fault-preserved?")]
        self.assertEqual(output.count("(e3-test-u32-ref "), 4)
        self.assertEqual(output.count("(e3-fault-counter-raw native "), 2)
        self.assertNotIn("e3-counter-ref", output)

    def test_balanced_closed_source(self) -> None:
        source = (SOURCE / "fault_driver.esk").read_text()
        depth = 0
        in_string = False
        escaped = False
        for line in source.splitlines():
            for char in line:
                if not in_string and char == ";":
                    break
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    self.assertGreaterEqual(depth, 0)
        self.assertEqual(depth, 0)
        self.assertFalse(in_string)


if __name__ == "__main__":
    unittest.main()
