"""Check the accepted shared source boundary, independently of its native tests."""
from pathlib import Path
import contextlib
import copy
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


def _render(form):
    if isinstance(form, list):
        return "(" + " ".join(_render(item) for item in form) + ")"
    return form


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


def validate_m3_call_guard(source):
    """Require reentry check, then cleanup installation, then busy publication."""
    macro = next(row for row in forms(source)
                 if row[:2] == ["define-syntax", "m3-call"])
    boundary = macro[2][2][1]
    if boundary[:2] != ["m3t-boundary", "operation"] or len(boundary) != 4:
        raise ValueError("m3-call must have one boundary, reentry check and guard")
    reentry, cleanup = boundary[2:]
    if reentry[:2] != ["if", ["vector-ref", "m3-call-state", "0"]]:
        raise ValueError("m3-call reentry check must precede cleanup")
    expected_rejection = ["m3t-fail", "'invalid-state", "operation", '"model',
                          "operation", "is", "already", 'active"']
    if len(reentry) != 3 or reentry[2] != expected_rejection:
        raise ValueError("m3-call must reject reentry with its fixed operation error")
    if cleanup[0] != "guard" or len(cleanup) != 4:
        raise ValueError("m3-call must install one cleanup guard around publication and body")
    handler, publish, body = cleanup[1:]
    clear = ["vector-set!", "m3-call-state", "0", "#f"]
    if publish != ["vector-set!", "m3-call-state", "0", "#t"]:
        raise ValueError("m3-call must publish busy only inside the installed cleanup guard")
    if handler != ["caught", ["#t", ["begin", clear,
                                      ["m3t-rethrow-raw", "caught", "operation"]]]]:
        raise ValueError("m3-call exceptional cleanup must clear then rethrow")
    expected_body = ["let", [["answer", ["begin", "body", "..."]]], clear, "answer"]
    if body != expected_body:
        raise ValueError("m3-call normal cleanup must clear before returning")
    return boundary


def validate_emergency_rethrow_entry(source):
    """Require every m3t-rethrow invocation to enter the compiler bridge."""
    definition = next(row for row in forms(source)
                      if row[:2] == ["define", ["m3t-rethrow-raw", "caught", "operation"]])
    if definition[-2:] != [":runtime-emergency-rethrow-param", "caught"]:
        raise ValueError("m3t-rethrow-raw must canonicalize caught at physical entry")
    if len(definition) != 5:
        raise ValueError("m3t-rethrow-raw must have one body and one entry modifier")
    return definition


class SharedContract(unittest.TestCase):
    def test_m3t_rethrow_canonicalizes_at_every_physical_entry(self):
        source = (ROOT / "native/m3t_transport_extension.esk").read_text()
        definition = validate_emergency_rethrow_entry(source)
        for mutation in ("drop-modifier", "wrong-formal", "move-before-body"):
            changed = copy.deepcopy(definition)
            if mutation == "drop-modifier":
                changed = changed[:-2]
            elif mutation == "wrong-formal":
                changed[-1] = "operation"
            else:
                changed[-3:] = changed[-2:] + changed[-3:-2]
            mutated = copy.deepcopy(forms(source))
            index = next(i for i, row in enumerate(mutated) if row == definition)
            mutated[index] = changed
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_emergency_rethrow_entry(" ".join(_render(row) for row in mutated))

    def test_m3_call_installs_cleanup_before_busy_publish(self):
        source = (ROOT / "native/m3_model_extension.esk").read_text()
        boundary = validate_m3_call_guard(source)
        for mutation in ("publish-before-guard", "missing-exception-clear",
                         "missing-normal-clear", "reentry-after-guard",
                         "dropped-reentry-rejection", "body-binding-bypass"):
            changed = copy.deepcopy(boundary)
            guard = changed[3]
            if mutation == "publish-before-guard":
                changed[3:] = [guard[2], [guard[0], guard[1], guard[3]]]
            elif mutation == "missing-exception-clear":
                guard[1][1][1].pop(1)
            elif mutation == "missing-normal-clear":
                guard[3].pop(-2)
            elif mutation == "reentry-after-guard":
                changed[2], changed[3] = changed[3], changed[2]
            elif mutation == "dropped-reentry-rejection":
                changed[2][2] = "#f"
            else:
                guard[3][1][0][1] = ["begin", "#t"]
            mutated = copy.deepcopy(forms(source))
            macro = next(row for row in mutated if row[:2] == ["define-syntax", "m3-call"])
            macro[2][2][1] = changed
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_m3_call_guard(" ".join(_render(row) for row in mutated))

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

    def test_retired_controls_use_one_intrusive_interval_index(self):
        f32 = (ROOT / "native/f32_tensor.c").read_text()
        integration = (ROOT / "src/eshkol_transformer/m3_call_f32_integration.c").read_text()
        reference = f32.split("static int storage_aliases_live_reference", 1)[1].split(
            "static int storage_aliases_live", 1)[0]
        foreign = integration.split("static int m3_call_foreign_storage", 1)[1].split(
            "static int m3_call_add_span", 1)[0]
        query = f32.split("static int f32_retired_control_overlaps", 1)[1].split(
            "static int aligned_pointer", 1)[0]
        retired_lists = ("retired_tensors", "retired_borrows", "retired_copy_plans",
                         "retired_parameters", "retired_gradient_plans",
                         "retired_reset_plans")
        self.assertEqual(f32.count("f32_retired_index_insert("), 7)
        self.assertLess(query.index("bytes == 0u"), query.index("!pointer_span_fits"))
        self.assertLess(query.index("!pointer_span_fits"), query.index("node == NULL"))
        self.assertIn("return f32_retired_control_overlaps(storage, bytes);", reference)
        self.assertIn("return f32_retired_control_overlaps(p, bytes);", foreign)
        for name in retired_lists:
            self.assertNotIn(name, reference)
            self.assertNotIn(name, foreign)
        for size in (88, 96, 40, 64, 48, 40):
            self.assertIn(f"== {size}u", f32)

    def test_retired_index_failstop_dependency_is_exact(self):
        for name in ("f32_tensor_undefined_symbols.txt",
                     "i2_wave2_undefined_symbols.txt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                inventory = ROOT / "native" / name
                symbols = inventory.read_text().splitlines()
                self.assertEqual(symbols, sorted(set(symbols)))
                self.assertIn("abort", symbols)
                without_abort = Path(directory) / "without-abort.txt"
                without_abort.write_text(
                    "\n".join(s for s in symbols if s != "abort") + "\n")
                result = subprocess.run(
                    ["cmp", without_abort, inventory], capture_output=True)
                self.assertEqual(result.returncode, 1)

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
