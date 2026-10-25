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
    owner = files["src/eshkol_transformer/g3c4_model_owner.c"]
    initializer = files["src/eshkol_transformer/m3t_transport.c"]
    module = files["internal/p1/lib/transformer/module.esk"]

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
    rank_two = [tuple(map(int, re.findall(r"\d+", row))) for row in shape_rows]
    rank_two = [shape for shape in rank_two if len(shape) == 2]
    assert len(rank_two) == 8
    assert sum(a * b for a, b in rank_two) == 1168
    assert "(define g3c4-model-cleanup-root (vector #f))" in c4
    assert "(define (g3c4-baseline-tokenizer-entry" in call
    assert "(t1-private-entry tokenizer)" in call
    assert "sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704" in call
    assert "(define (i2-g3c4-construction-prepare-train-internal! value)" in i2
    assert "(vector-ref p1-trusted-surface 73)" in p1
    for operation in ("begin", "check", "end"):
        assert f"et_g3c4_model_pins_{operation}_internal" in pins
    assert "(define m3t-algorithm 'n3k.philox4x32-10.uniform-f32.v1)" in m3t
    assert "(m3t-entry value 'initializer 'diagnostic-initializer-words #t)" in m3t
    assert "(m3t-native-initializer-word native 3)" in m3t
    assert "s->words[0]=1;s->words[1]=seed;" in initializer
    assert "s->words[i]" in initializer
    assert "owner->original[0] = 1;" in owner
    assert "owner->original[1] = seed;" in owner
    assert "memcpy(owner->successor, owner->original, sizeof(owner->original));" in owner
    assert "owner->successor[2] != 292 || owner->successor[3] != 0" in owner
    assert "(define (module-parameters module)\n  (commit-module-observation!" in module
    assert "(install-canonical-paths! (vector-ref observation 1))" in module
    assert "(finalize-modules! (vector-ref observation 2))" in module


def asserted_contract(contract: str) -> None:
    assert "The next gate is constructor-only" in contract
    assert "Any broader raw V256 admission needs a later generation-contract change" in contract
    assert "Slot 4 is the symbol `'trainable`; slot 11 is the exact private\nprofile symbol `'trainable-c4`" in contract
    assert "**nine-slot vector**" in contract
    for index, field in enumerate((
        "`'g3c4-trainable-metadata-v1`",
        "resolved X1 config shell",
        "X1 canonical serialization",
        "X1 fingerprint",
        "raw V256 T1 tokenizer shell",
        "T1 fingerprint",
        "M3T initializer shell",
        "`'n3k.philox4x32-10.uniform-f32.v1`",
        "four-word vector",
    )):
        assert re.search(rf"^\| {index} \| {re.escape(field)} \|", contract, re.M)
    assert "`[1, seed, 0, 0]`" in contract
    assert "`[1, seed, 292, 0]`" in contract
    assert "1,168 f32 values\nin 292 four-value counter blocks" in contract
    assert "does\n**not** consume or advance the M3T initializer" in contract
    assert "must not call `module-parameters` or its\n`commit-module-observation!` path" in contract
    assert "The constructor gate does not create a workspace or call a workspace\npreflight" in contract


PATHS = (
    "native/g3c4_model_extension.esk",
    "native/g3c4_call_entry_extension.esk",
    "native/g3c4_i2_construction_extension.esk",
    "native/g3c4_p1_construction_extension.esk",
    "native/m3t_transport_extension.esk",
    "src/eshkol_transformer/m3_call_pins.h",
    "src/eshkol_transformer/g3c4_model_owner.c",
    "src/eshkol_transformer/m3t_transport.c",
    "internal/p1/lib/transformer/module.esk",
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
        asserted_contract(contract)
        for marker in (
            "Proposed private contract",
            "C4 train-mode prepare",
            "same C4 registry",
            "Separate future workspace gate",
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

    def test_changed_native_rng_transition_is_detected(self) -> None:
        changed = dict(self.files)
        changed["src/eshkol_transformer/g3c4_model_owner.c"] = changed[
            "src/eshkol_transformer/g3c4_model_owner.c"
        ].replace("owner->successor[2] != 292", "owner->successor[2] != 293", 1)
        with self.assertRaises(AssertionError):
            asserted_anchors(changed)

    def test_changed_metadata_layout_is_detected(self) -> None:
        contract = source("docs/g3/G3_R_C4_TRAINABLE_CONSTRUCTOR_WORKSPACE_AUTHORITY_CONTRACT.md")
        with self.assertRaises(AssertionError):
            asserted_contract(contract.replace("| 8 | four-word vector |", "| 8 | seed only |", 1))

    def test_workspace_mutation_route_is_detected(self) -> None:
        changed = dict(self.files)
        changed["internal/p1/lib/transformer/module.esk"] = changed[
            "internal/p1/lib/transformer/module.esk"
        ].replace("(install-canonical-paths! (vector-ref observation 1))", "", 1)
        with self.assertRaises(AssertionError):
            asserted_anchors(changed)

    def test_contract_forbids_module_parameters_preflight(self) -> None:
        contract = source("docs/g3/G3_R_C4_TRAINABLE_CONSTRUCTOR_WORKSPACE_AUTHORITY_CONTRACT.md")
        with self.assertRaises(AssertionError):
            asserted_contract(contract.replace("must not call `module-parameters`", "may call `module-parameters`", 1))


if __name__ == "__main__":
    unittest.main()
