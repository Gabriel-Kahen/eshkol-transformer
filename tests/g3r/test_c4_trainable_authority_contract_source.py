#!/usr/bin/env python3
"""Check the source boundary assumed by the proposed C4 authority contract."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


def source(path: str) -> str:
    return (ROOT / path).read_text()


def asserted_anchors(files: dict[str, str]) -> None:
    c4 = files["native/g3c4_model_extension.esk"]
    call = files["native/g3c4_call_entry_extension.esk"]
    i2 = files["native/g3c4_i2_construction_extension.esk"]
    p1 = files["native/g3c4_p1_construction_extension.esk"]
    m3t = files["native/m3t_transport_extension.esk"]
    pins = files["src/eshkol_transformer/m3_call_pins.h"]

    assert "(define g3c4-model-registry (vector '()))" in c4
    assert "(define (g3c4-model-create-seeded-internal seed)" in c4
    assert "(vector #f 'c4-model 'pending #f 'seeded" in c4
    assert "original successor #f handles #f #f" in c4
    assert "(vector-ref canonical-handles 10)" in c4
    assert "(i2-g3c4-construction-prepare-eval-internal!" in c4
    assert "(g3c4-native-owner-create-seeded seed)" in c4
    assert "(i2-native-g3c4-owner-bind" in i2
    assert "(eq? (vector-ref entry 4) 'seeded)" in call
    assert "(eq? (module-mode-internal (vector-ref model-entry 0)) 'eval)" in call
    assert "(define (i2-g3c4-construction-begin-internal native-owner)" in i2
    assert "(define (i2-g3c4-construction-prepare-eval-internal! value)" in i2
    assert "(define (i2-g3c4-construction-seal-prepared-internal! value)" in i2
    assert "(vector-ref p1-trusted-surface 71)" in p1
    assert "(vector-ref p1-trusted-surface 72)" in p1
    assert re.search(r"\(define m3t-shapes\s+'#\(.*?\(2 4\)\)\)", m3t, re.S)
    assert "((model.context-length . 2)" in m3t
    c4_shapes = c4.split("(define g3c4-model-shapes", 1)[1].split("(define (", 1)[0]
    shape_rows = re.findall(r"\([^()]*\)", c4_shapes)
    assert len(shape_rows) == 14 and shape_rows[-1] == "(4 4)"
    assert "(define g3c4-model-cleanup-root (vector #f))" in c4
    assert "(define (g3c4-baseline-tokenizer-entry" in call
    assert "(t1-private-entry tokenizer)" in call
    assert "(define (i2-g3c4-construction-prepare-train-internal! value)" in i2
    assert "(vector-ref p1-trusted-surface 73)" in p1
    for operation in ("begin", "check", "end"):
        assert f"et_g3c4_model_pins_{operation}_internal" in pins


PATHS = (
    "native/g3c4_model_extension.esk",
    "native/g3c4_call_entry_extension.esk",
    "native/g3c4_i2_construction_extension.esk",
    "native/g3c4_p1_construction_extension.esk",
    "native/m3t_transport_extension.esk",
    "src/eshkol_transformer/m3_call_pins.h",
)


class ContractSourceBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.files = {path: source(path) for path in PATHS}

    def test_current_boundary(self) -> None:
        asserted_anchors(self.files)
        contract = source(
            "docs/g3/G3_R_C4_TRAINABLE_CONSTRUCTOR_WORKSPACE_AUTHORITY_CONTRACT.md"
        )
        for marker in (
            "Proposed private contract",
            "C4 train-mode prepare",
            "same C4 registry",
            "identity and shape only",
            "same live entry and owner in place",
        ):
            self.assertIn(marker.lower(), contract.lower())

    def test_mutated_m3t_profile_is_detected(self) -> None:
        changed = dict(self.files)
        changed["native/m3t_transport_extension.esk"] = changed[
            "native/m3t_transport_extension.esk"
        ].replace("((model.context-length . 2)", "((model.context-length . 4)", 1)
        with self.assertRaises(AssertionError):
            asserted_anchors(changed)

    def test_mutated_c4_authority_is_detected(self) -> None:
        changed = dict(self.files)
        changed["native/g3c4_model_extension.esk"] = changed[
            "native/g3c4_model_extension.esk"
        ].replace("'pending #f 'seeded", "'pending #f 'trainable", 1)
        with self.assertRaises(AssertionError):
            asserted_anchors(changed)

    def test_missing_prepared_route_is_detected(self) -> None:
        changed = dict(self.files)
        changed["native/g3c4_i2_construction_extension.esk"] = changed[
            "native/g3c4_i2_construction_extension.esk"
        ].replace("(define (i2-g3c4-construction-prepare-eval-internal! value)", "", 1)
        with self.assertRaises(AssertionError):
            asserted_anchors(changed)

    def test_changed_c4_position_shape_is_detected(self) -> None:
        changed = dict(self.files)
        changed["native/g3c4_model_extension.esk"] = changed[
            "native/g3c4_model_extension.esk"
        ].replace("(256 4) (4) (4) (4 4)))", "(256 4) (4) (4) (2 4)))", 1)
        with self.assertRaises(AssertionError):
            asserted_anchors(changed)


if __name__ == "__main__":
    unittest.main()
