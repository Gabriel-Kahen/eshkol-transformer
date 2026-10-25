"""Pin the source basis for the proposed C4 workspace topology authority."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


def source(path: str) -> str:
    return (ROOT / path).read_text()


def c4_shape_pair(model: str, native: str) -> bool:
    model_shapes = model.split("(define g3c4-model-shapes", 1)[1].split(
        "(define (g3c4-fail-raw", 1
    )[0]
    native_shapes = native.split("static const uint64_t g3c4_call_shapes[14][2] = {", 1)[1].split(
        "};", 1
    )[0]
    return bool(re.search(r'\("position_embedding" "weight"\)', model)) and bool(
        re.search(r'\(256 4\).*\(4\)\s+\(4\)\s+\(4 4\)', model_shapes, re.S)
    ) and bool(re.search(r'\{256,4\},\s*\{4,0\},\s*\{4,0\},\s*\{4,4\}', native_shapes))


class WorkspaceTopologyContractSource(unittest.TestCase):
    def test_constructor_roots_exact_trainable_owner_and_handles(self) -> None:
        text = source("native/g3c4_trainable_constructor_extension.esk")
        for phrase in (
            "(vector #f 'c4-model 'pending #f 'trainable",
            "g3c4-trainable-model-profile))",
            "(handles (make-vector 14 #f))",
            "(vector-set! g3c4-model-registry 0 next)",
            "(i2-native-g3c4-owner-parameter owner index)",
            "(vector-ref canonical-handles 10)",
            "(i2-g3c4-construction-prepare-train-internal!",
            "(vector-set! canonical-entry 2 'live)",
        ):
            self.assertIn(phrase, text)

    def test_c4_shape_differs_from_tr3_fixed_profile(self) -> None:
        model = source("native/g3c4_model_extension.esk")
        native = source("src/eshkol_transformer/m3_call_f32_integration.c")
        p1 = source("internal/p1/lib/transformer/module.esk")
        self.assertTrue(c4_shape_pair(model, native))
        self.assertIn("(tr3-fixed-dimension=? (car shape) 2)", p1)
        self.assertIn("(= index 13)", p1)
        self.assertFalse(c4_shape_pair(model.replace("(4 4)))", "(2 4)))"), native))
        self.assertFalse(c4_shape_pair(model, native.replace("{4,0}, {4,0}, {4,4}", "{4,0}, {4,0}, {2,4}")))

    def test_module_parameters_is_mutating_observation(self) -> None:
        p1 = source("internal/p1/lib/transformer/module.esk")
        self.assertIn("(commit-module-observation! (prepare-module-parameters module))", p1)
        self.assertIn("(vector-ref p1-trusted-surface 73)",
                      source("native/g3c4_p1_construction_extension.esk"))
        self.assertNotIn("(vector-ref p1-trusted-surface 74)", p1)

    def test_native_live_status_is_new_dependency(self) -> None:
        owner = source("src/eshkol_transformer/g3c4_model_owner.c")
        pins = source("src/eshkol_transformer/m3_call_f32_integration.c")
        self.assertIn("owner != staged_c4_owner", owner)
        self.assertIn("et_g3c4_construction_parameter_preflight_internal", owner)
        for name in ("begin", "check", "end"):
            self.assertIn(f"et_g3c4_model_pins_{name}_internal", pins)
        self.assertIn("owner->handles[found] != exact_handle", owner)
        self.assertNotIn("et_g3c4_private_model_owner_topology_status_v1", owner)

    def test_existing_lease_predicate_has_fixed_registry_scope(self) -> None:
        lease = source("native/tr3_lease_core_extension.esk")
        self.assertIn("(define (tr3-o2-create-overlap-check-internal handles)", lease)
        self.assertIn("(vector-ref tr3-trainers 0)", lease)
        self.assertIn("(vector-ref (vector-ref (car records) 11) index)", lease)

    def test_contract_does_not_claim_workspace_admission(self) -> None:
        contract = source("docs/g3/G3_R_C4_WORKSPACE_TOPOLOGY_AUTHORITY_CONTRACT.md")
        for phrase in (
            "source contract only", "slot **74**", "module-parameters",
            "final position `[4,4]`", "swapped P1 handle",
            "owner-to-P1 association proof", "topology and owner identity only",
            "active generation/workspace/lease exclusion",
        ):
            self.assertIn(phrase, contract)


if __name__ == "__main__":
    unittest.main()
