"""Source contract for the private, checked C4 P3 terminal route."""

from pathlib import Path
import unittest

from tests.g3c4.test_public_g2_port_source import forms, nodes


SOURCE = Path(__file__).resolve().parents[2] / (
    "native/g3c4_p2g2_terminal_commit_extension.esk")


def valid(source):
    try:
        tree = forms(source)
        core = next(node for node in nodes(tree)
                    if len(node) == 2 and node[0] == "core"
                    and node[1][:2] == [
                        "lambda",
                        ["generator", "input", "enrollment", "resume-ticket"]])
        issuer = [node for node in nodes(core)
                  if node and node[0] == "cpu4-c4-p3-resume-issuer-internal"]
        assert [node[1] for node in issuer] == ["'admit", "'verify"]
        assert all(node[2:5] == ["generator", "input", "resume-ticket"]
                   for node in issuer)
        assert ["eq?", "resume-descriptor", issuer[1]] in list(nodes(core))
        route = [node for node in nodes(tree)
                 if len(node) == 3 and node[:2] == [
                     "set!", "g3c4-p2g2-terminal-resume-private!"]]
        assert len(route) == 1
        assert ["core", "generator", "input", "#f", "ticket"] in list(
            nodes(route[0]))
        assert ["lambda", ["generator", "input"],
                ["core", "generator", "input", "#f", "#f"]] in list(nodes(tree))
        required = (
            "(= (bytevector-length (vector-ref descriptor 0)) 24)",
            "(= (bytevector-length (vector-ref descriptor 1)) 32)",
            "(g3c4-p2g2-resume-word? (vector-ref descriptor 0) 8 197)",
            "(g3c4-p2g2-resume-word? (vector-ref descriptor 1) 8 41)",
            "(g3c4-p2g2-terminal-pending-reserve! call)",
            "(g3c4-p2g2-terminal-live-prepare call pending)",
            "(g3c4-p2g2-native-p3-resume",
            "(t1-private-g3-decode-raw-into!",
            "(vector-set! protected-anchor 13",
            "(vector-set! protected-record 13",
            "(vector-set! (vector-ref protected-record 14)",
            "(g3c4-p2g2-terminal-second-sample-decode!",
            "(g3c4-p2g2-native-terminal-commit",
        )
        assert all(part in source for part in required)
        assert [source.index(part) for part in required[4:]] == sorted(
            source.index(part) for part in required[4:])
        assert "(if (not resume)\n                             " + (
            "(g3c4-p2g2-terminal-non-eos-prefix-commit!") in source
        assert "(g3c4-p2g2-terminal-first-sample-decode! call pending)" in source
        assert "(if (not (eq? resume-descriptor" in source
        assert not any(node and node[0] == "procedure?" and
                       "resume-ticket" in node for node in nodes(core))
        return required
    except (AssertionError, ValueError, IndexError, StopIteration):
        return False


class P3TerminalResumeSource(unittest.TestCase):
    def test_checked_route_and_controls(self):
        source = SOURCE.read_text()
        required = valid(source)
        self.assertTrue(required)
        for part in required[:4] + required[6:11]:
            with self.subTest(part=part):
                self.assertFalse(valid(source.replace(part, "#f", 1)))
        self.assertFalse(valid(source.replace(
            "'verify generator input resume-ticket",
            "'other generator input resume-ticket", 1)))
        self.assertFalse(valid(source.replace(
            "(if (not (eq? resume-descriptor",
            "(if (not (equal? resume-descriptor", 1)))


if __name__ == "__main__":
    unittest.main()
