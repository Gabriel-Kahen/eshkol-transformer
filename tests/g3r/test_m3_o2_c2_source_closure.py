"""Static dependency check for the noninstalled joint AOT fixture."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/g3r/m3_o2_c2_joint_runtime.esk"
ALIAS = "(define t1-wave1-tensor-admitted? t2-wave2-tensor-admitted?)"
D2_LOAD = '(load "d2_wave2_root.esk")'
M3_LOAD = '(load "m3t_transport_extension.esk")'


def check_closure(source: str) -> None:
    positions = [source.find(item) for item in (D2_LOAD, ALIAS, M3_LOAD)]
    if -1 in positions or positions != sorted(positions) or len(set(positions)) != 3:
        raise ValueError("D2/T2 admission binding must precede M3 transport")
    if '(load "t1_wave1_root.esk")' in source:
        raise ValueError("joint fixture must not introduce a second T1 registry")


class JointSourceClosure(unittest.TestCase):
    def test_exact_upstream_predicate_and_order(self):
        source = FIXTURE.read_text()
        check_closure(source)
        d2 = (ROOT / "native/d2_wave2_root.esk").read_text()
        t2 = (ROOT / "native/t2_wave2_root.esk").read_text()
        m3 = (ROOT / "native/m3t_transport_extension.esk").read_text()
        self.assertIn('(load "t2_wave2_root.esk")', d2)
        self.assertIn('(define (t2-wave2-tensor-admitted? value)', t2)
        self.assertIn('(t1-wave1-tensor-admitted? tokens)', m3)

    def test_missing_or_late_binding_rejected(self):
        source = FIXTURE.read_text()
        for bad in (source.replace(ALIAS, "", 1),
                    source.replace(ALIAS + "\n" + M3_LOAD,
                                   M3_LOAD + "\n" + ALIAS, 1),
                    source.replace(ALIAS, '(load "t1_wave1_root.esk")\n' + ALIAS, 1)):
            with self.subTest(bad=bad.find(ALIAS)), self.assertRaises(ValueError):
                check_closure(bad)


if __name__ == "__main__":
    unittest.main()
