#!/usr/bin/env python3
"""Pin the proposed C4 constructor to existing private authorities."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "native/g3c4_trainable_constructor_extension.esk"


def source(path: str) -> str:
    return (ROOT / path).read_text()


def balanced_esk(text: str) -> bool:
    depth = 0
    quoted = False
    escaped = False
    comment = False
    for char in text:
        if comment:
            if char == "\n":
                comment = False
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == ";":
            comment = True
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not quoted


def asserted_constructor(text: str, model: str, i2: str, call: str) -> None:
    assert balanced_esk(text)
    assert "(define g3c4-trainable-model-profile 'trainable-c4)" in text
    assert "(define (g3c4-model-create-trainable-internal config tokenizer initializer)" in text
    assert "(x1-private-config-validate config)" in text
    assert "(model.context-length . 4)" in text
    assert "(g3c4-trainable-tokenizer-entry tokenizer operation)" in text
    assert "(t1-private-entry tokenizer)" in text
    assert "(m3t-entry initializer 'initializer operation #t)" in text
    assert "(eq? algorithm m3t-algorithm)" in text
    assert "(equal? words (list 1 seed 0 0))" in text
    assert "(string-copy (x1-private-config-canonical config))" in text
    assert "(string-copy (x1-private-config-fingerprint config))" in text
    assert "(string-copy (vector-ref t1-core 6))" in text
    assert "(vector 1 seed 0 0)" in text
    assert "(vector #f 'c4-model 'pending #f 'trainable" in text
    assert "original successor metadata handles #f #f" in text
    assert "g3c4-trainable-model-profile" in text
    assert "(g3c4-native-owner-create-seeded seed)" in text
    assert "(i2-g3c4-construction-prepare-train-internal!" in text
    assert "(g3c4-native-owner-prepare-seal owner)" in text
    assert "(g3c4-trainable-tail-preflight" in text
    assert "(i2-g3c4-construction-seal-prepared-internal! construction)" in text
    assert "(g3c4-native-owner-commit-seal owner)" in text
    assert "(g3c4-model-dead! entry)" in text
    assert "(i2-g3c4-construction-abort-internal!" in text
    assert "(g3c4-native-owner-abort" in text
    assert "(equal? successor\n                       (vector 1 (vector-ref original 1) 292 0))" in text
    assert "(equal? original (vector-ref metadata 8))" in text
    assert "(eq? (vector-ref metadata 0) 'g3c4-trainable-metadata-v1)" in text
    assert "(vector-ref g3c4-model-registry 0)" in text
    assert "(vector-ref g3c4-model-cleanup-root 0)" in text
    assert "(define g3c4-model-registry (vector '()))" in model
    assert "(define g3c4-model-cleanup-root (vector #f))" in model
    assert "(define (i2-g3c4-construction-prepare-train-internal! value)" in i2
    assert "(define (g3c4-baseline-tokenizer-entry" in call
    baseline = re.search(r'"sha256:eshkol-byte-tokenizer-v1:[a-f0-9]{64}"', call)
    assert baseline and baseline.group() in text
    assert "module-parameters" not in text
    assert "(define g3c4-model-registry" not in text
    assert "(define g3c4-model-cleanup-root" not in text


class TrainableConstructorSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = EXTENSION.read_text()
        cls.model = source("native/g3c4_model_extension.esk")
        cls.i2 = source("native/g3c4_i2_construction_extension.esk")
        cls.call = source("native/g3c4_call_entry_extension.esk")

    def test_source_boundary(self) -> None:
        asserted_constructor(self.text, self.model, self.i2, self.call)

    def test_fixture_has_same_registry_positive_and_typed_negatives(self) -> None:
        fixture = source("tests/g3c4/trainable_constructor_test.esk")
        self.assertTrue(balanced_esk(fixture))
        self.assertIn('(load "g3c4_trainable_constructor_extension.esk")', fixture)
        self.assertIn("(eq? (module-mode-internal model) 'train)", fixture)
        self.assertIn("(equal? (vector-ref entry 6) '#(1 1729 292 0))", fixture)
        for marker in (
            '"context two" \'unsupported',
            '"wrong initializer seed" \'invalid-argument',
            '"foreign tokenizer" \'invalid-argument',
            '"foreign initializer" \'invalid-argument',
            '"native allocation rollback" \'internal',
        ):
            self.assertIn(marker, fixture)
        self.assertNotIn("module-parameters", fixture)

    def test_wrong_prepare_route_is_detected(self) -> None:
        mutant = self.text.replace(
            "(i2-g3c4-construction-prepare-train-internal!",
            "(i2-g3c4-construction-prepare-eval-internal!", 1,
        )
        with self.assertRaises(AssertionError):
            asserted_constructor(mutant, self.model, self.i2, self.call)

    def test_wrong_rng_successor_is_detected(self) -> None:
        mutant = self.text.replace("(vector 1 (vector-ref original 1) 292 0)",
                                   "(vector 1 (vector-ref original 1) 290 0)", 1)
        with self.assertRaises(AssertionError):
            asserted_constructor(mutant, self.model, self.i2, self.call)

    def test_alternate_registry_is_detected(self) -> None:
        mutant = self.text.replace("(vector-ref g3c4-model-registry 0)",
                                   "(vector-ref trainable-registry 0)")
        with self.assertRaises(AssertionError):
            asserted_constructor(mutant, self.model, self.i2, self.call)

    def test_mutating_workspace_read_is_detected(self) -> None:
        with self.assertRaises(AssertionError):
            asserted_constructor(self.text + "\n(module-parameters model)\n",
                                 self.model, self.i2, self.call)

    def test_surplus_parenthesis_is_detected(self) -> None:
        with self.assertRaises(AssertionError):
            asserted_constructor(self.text + ")", self.model, self.i2, self.call)


if __name__ == "__main__":
    unittest.main()
