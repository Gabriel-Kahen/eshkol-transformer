"""Cheap source-contract checks; this does not execute the Eshkol runtime."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/cpu4_trainer_control_extension.esk"


def parse(text):
    tokens = re.findall(r';[^\n]*|"(?:\\.|[^"\\])*"|[()\']|[^\s()\';]+', text)
    tokens = [token for token in tokens if not token.startswith(";")]
    index = 0
    def datum():
        nonlocal index
        token = tokens[index]
        index += 1
        if token == "(":
            value = []
            while tokens[index] != ")":
                value.append(datum())
            index += 1
            return value
        if token == "'":
            return ["quote", datum()]
        if token == ")":
            raise ValueError("unmatched close parenthesis")
        return json.loads(token) if token.startswith('"') else token
    forms = []
    while index < len(tokens):
        forms.append(datum())
    return forms


def walk(value):
    if isinstance(value, list):
        yield value
        for item in value:
            yield from walk(item)


class SourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text()
        cls.forms = parse(cls.text)
        cls.functions = {form[1][0]: form for form in cls.forms
                         if len(form) > 2 and form[0] == "define" and isinstance(form[1], list)}

    def test_own_record_schema_and_initial_reproducibility(self):
        form = self.functions["cpu4-trainer-create-internal"]
        record = next(node for node in walk(form)
                      if len(node) > 1 and node[:2] == ["vector", "cpu4-trainer-tag"])[1:]
        self.assertEqual(len(record), 22)
        self.assertEqual(record[1], "#f")
        self.assertEqual(record[3], ["quote", "idle"])
        self.assertEqual(record[4], "#f")
        self.assertEqual(record[6], "resolved")
        self.assertEqual(record[9], "binding-token")
        self.assertEqual(record[10:13], [["vector-ref", "binding", str(i)] for i in (9, 10, 11)])
        self.assertEqual(record[13:18], ["optimizer", "o2", "dataset", "state", "cursor"])
        self.assertEqual(record[18][:3], ["vector", ["quote", "philox4x32-10"], "1"])
        self.assertEqual(record[18][-2:], ["0", "0"])
        self.assertEqual(record[19:], ["0", "0", "0"])

    def test_no_old_trainer_authority_or_bare_extern_capture(self):
        self.assertNotIn("tr3-lease-", self.text)
        self.assertNotIn("tr3-trainers", self.text)
        self.assertNotIn("m3t-entry", self.text)
        externs = {form[2] for form in self.forms if form[0] == "extern"}
        self.assertEqual(externs, {"cpu4-control-native-enter", "cpu4-control-native-leave"})
        for form in self.forms:
            if form[0] == "extern":
                continue
            for node in walk(form):
                self.assertNotEqual(node[:1], ["length"])
                for index, item in enumerate(node):
                    if isinstance(item, str) and item in externs:
                        self.assertEqual(index, 0, "extern used as a callable value")

    def test_snapshot_restore_use_actual_write_lease(self):
        begin = self.functions["cpu4-trainer-control-begin-internal"]
        nodes = list(walk(begin))
        self.assertIn(["cpu4-control-native-enter", ["vector-ref", "canonical", "6"], "2"], nodes)
        self.assertIn(["eq?", "kind", ["quote", "step"]], nodes)
        self.assertTrue(any(node[:2] == ["cpu4-control-fail", ["quote", "unsupported"]] for node in nodes))
        end = list(walk(self.functions["cpu4-trainer-control-end-internal!"]))
        self.assertIn(["cpu4-control-native-leave", ["vector-ref", "parent", "6"]], end)

    def test_restore_tail_changes_only_five_canonical_slots(self):
        form = self.functions["cpu4-trainer-controls-restore-commit-internal!"]
        writes = [node for node in walk(form) if len(node) > 2 and node[:2] == ["vector-set!", "record"]]
        self.assertEqual([node[2] for node in writes], [str(i) for i in range(17, 22)])
        tail = form[2][2:]
        first = next(i for i, node in enumerate(tail) if isinstance(node, list) and node[:2] == ["vector-set!", "record"])
        self.assertTrue(all(node == "#t" or node[0] == "vector-set!" for node in tail[first:]))
        for statement in tail[first:]:
            for node in walk(statement):
                if node and node[0] == "quote":
                    self.assertEqual(node, ["quote", "committed"])
                elif node:
                    self.assertIn(node[0], ("vector-set!", "vector-ref"))

    def test_controls_validate_exact_cursor_config_and_contribution_equation(self):
        form = self.functions["cpu4-control-validate-copy"]
        nodes = list(walk(form))
        self.assertIn(["c2-d2-cursor-pair-internal", "request"], nodes)
        self.assertIn(["d2-core-state-identity=?", "core", "decoded"], nodes)
        self.assertIn(["=", ["vector-ref", "decoded", "5"], "4"], nodes)
        self.assertIn(["equal?", ["vector-ref", "config", "0"], "canonical"], nodes)
        self.assertIn(["cpu4-control-product", "updates", "accumulation", "operation"], nodes)
        self.assertIn(["cpu4-control-product", "4", "contributions", "operation"], nodes)
        self.assertIn(["=", "contributions", ["+", "epoch-contributions", "offset"]], nodes)
        self.assertIn([">", "contributions", "tokens"], nodes)
        self.assertIn([">", "tokens", "maximum-tokens"], nodes)

    def test_actual_component_dependencies_exist(self):
        dependencies = {
            "native/cpu4_model_binding_extension.esk": ["cpu4-model-binding-check-internal", "cpu4-d2-idle-core"],
            "native/o2_wave2_extension.esk": ["o2-require-optimizer", "o2-native-optimizer-require-absent-gradients"],
            "native/c2_d2_cursor_pair_extension.esk": ["c2-d2-cursor-pair-internal", "c2-d2-decode"],
            "src/eshkol_transformer/g3c4_model_owner_internal.h": ["et_g3c4_private_training_lease_enter_v1", "et_g3c4_private_training_lease_leave_v1"],
        }
        for path, names in dependencies.items():
            content = (ROOT / path).read_text()
            for name in names:
                self.assertIn(name, content)


if __name__ == "__main__":
    unittest.main()
