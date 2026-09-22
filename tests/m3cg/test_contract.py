"""Check the accepted shared source boundary, independently of its native tests."""
from pathlib import Path
import contextlib
import io
import json
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tests.m3cg import measure_gate

ROOT = Path(__file__).resolve().parents[2]


def forms(text):
    tokens = iter(re.findall(r"[()]|[^\s()]+", re.sub(r";[^\n]*", "", text)))
    def one(token):
        if token != "(":
            if token == ")":
                raise ValueError("unmatched close")
            return token
        result = []
        for token in tokens:
            if token == ")":
                return result
            result.append(one(token))
        raise ValueError("unclosed form")
    return [one(token) for token in tokens]


def expected_adapters():
    source = (ROOT / "native/m3t_transport_extension.esk").read_text()
    definitions = {name: args.split() for name, args in re.findall(
        r"\(define \((m3t-public-[^\s()]+)([^()]*)\)", source)}
    expected = []
    for row in (ROOT / "native/m3_package_private_renames.txt").read_text().splitlines():
        name, _ = row.split()
        if name.startswith("m3t-public-"):
            suffix = name.removeprefix("m3t-public-")
            args = definitions[name]
            expected.append(["define", ["g3t-checked-m3t-" + suffix, *args],
                             ["m3-call", "'diagnostic-" + suffix, [name, *args]]])
    return expected


class SharedContract(unittest.TestCase):
    def test_measurement_executes_both_gates_and_preserves_failure(self):
        expected = [["/usr/bin/bash", str(ROOT / "scripts" / name)] for name in (
            "test-m3cg-native.sh", "test-m3cg-package.sh")]
        for codes in ((0, 0), (7,), (0, 9)):
            with self.subTest(codes=codes), tempfile.TemporaryDirectory() as directory:
                responses = [subprocess.CompletedProcess([], code) for code in codes]
                with patch.object(measure_gate.sys, "argv", ["measure_gate.py", directory]), \
                     patch.object(measure_gate.subprocess, "run", side_effect=responses) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(measure_gate.main(), codes[-1])
                self.assertEqual([call.args[0] for call in run.call_args_list], expected[:len(codes)])
                self.assertTrue(all(call.kwargs == {"cwd": ROOT} for call in run.call_args_list))
                report = json.loads((Path(directory) / "gate-measurements.json").read_text())
                self.assertEqual([row["exit_code"] for row in report["gates"]], list(codes))
                self.assertEqual([row["gate"] for row in report["gates"]],
                                 [Path(command[-1]).name for command in expected[:len(codes)]])
                self.assertGreaterEqual(report["seconds"], 0)

    def test_all38_exact_single_guard_adapters(self):
        observed = forms((ROOT / "native/m3_call_adapters.esk").read_text())
        expected = expected_adapters()
        self.assertEqual(len(expected), 38)
        self.assertEqual(observed, expected)
        self.assertEqual(len({row[1][0] for row in observed}), 38)

    def test_adapter_mutations_are_observable(self):
        expected = expected_adapters()
        source = (ROOT / "native/m3_call_adapters.esk").read_text()
        for before, after in (
            ("m3-call", "m3t-boundary"),
            ("'diagnostic-attention!", "'attention!"),
            ("(m3t-public-attention! workspace)", "(g3t-checked-m3t-attention! workspace)"),
            ("(g3t-checked-m3t-attention! workspace)", "(g3t-checked-m3t-attention! workspace ignored)"),
        ):
            with self.subTest(before=before):
                self.assertIn(before, source)
                self.assertNotEqual(forms(source.replace(before, after, 1)), expected)
        self.assertNotEqual(forms(source)[:-1], expected)
        self.assertNotEqual(forms(source) + forms(source)[:1], expected)

    def test_exact_header_contract_and_single_f32_owner(self):
        document = (ROOT / "docs/E3_SHARED_CALL_CONTRACT.md").read_text()
        declaration = re.search(r"```c\n(.*?)\n```", document, re.S)[1]
        header = (ROOT / "src/eshkol_transformer/m3_call_pins.h").read_text()
        def normalized(text):
            return re.sub(r"\s+", "", re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S))
        self.assertEqual(normalized(header), normalized(declaration))
        implementation = (ROOT / "src/eshkol_transformer/m3_call_f32_integration.c").read_text()
        includes = re.findall(r'^#include "([^"]+)"', implementation, re.M)
        self.assertEqual(includes, ["m3t_f32_integration.c", "m3_call_pins.h"])
        for forbidden in ("g3t_transport", "e3_frame", "m3_model.c", "m3-call-state"):
            self.assertNotIn(forbidden, implementation)

    def test_predecessor_source_bytes_and_public_boundary(self):
        result = subprocess.run(["sha256sum", "--quiet", "-c", "tests/m3cg/predecessor_sources.sha256"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for suffix, count in (("private_renames", 46), ("public_exports", 87),
                              ("defined_symbols", 93), ("facades", 7)):
            self.assertEqual(len((ROOT / f"native/m3_package_{suffix}.txt").read_text().splitlines()), count)
        for path in (ROOT / "lib").rglob("*.esk"):
            self.assertNotIn("m3_call_", path.read_text())
            self.assertNotIn("g3t-checked-m3t-", path.read_text())


if __name__ == "__main__":
    unittest.main()
