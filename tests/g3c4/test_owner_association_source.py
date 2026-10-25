"""Focused source guard for the uncompiled private association wrapper."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = (ROOT / "native/g3c4_trainable_owner_association_extension.esk").read_text()
FIXTURE = (ROOT / "tests/g3c4/owner_association_status_test.esk").read_text()
OWNER = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()


def balanced(source: str) -> bool:
    depth = 0
    quoted = False
    escaped = False
    for line in source.splitlines():
        for ch in line.split(";;;", 1)[0]:
            if quoted:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    quoted = False
            elif ch == '"':
                quoted = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False
    return depth == 0 and not quoted


class AssociationSource(unittest.TestCase):
    def test_closed_read_only_wrapper(self):
        self.assertTrue(balanced(WRAPPER))
        self.assertTrue(balanced(FIXTURE))
        for forbidden in ("(vector-set!", "(module-parameters", "(i2-carrier ",
                          "(module-construction-", "(g3c4-native-owner-bind"):
            self.assertNotIn(forbidden, WRAPPER)
        self.assertIn("(eq? owner (vector-ref entry 3))", WRAPPER)
        self.assertIn("(eq? (vector-ref entry 2) 'live)", WRAPPER)
        self.assertIn("(eq? (vector-ref entry 4) 'trainable)", WRAPPER)
        self.assertIn("(eq? (vector-ref entry 10) #f)", WRAPPER)

    def test_exact_p1_to_i2_to_native_chain(self):
        self.assertEqual(WRAPPER.count("(module-c4-train-topology-status-internal"), 2)
        self.assertEqual(WRAPPER.count("(g3r-c4-native-owner-topology-status"), 2)
        for required in (
            "(eq? (vector-ref handles index)",
            "(parameter-handle-tensor-identity-internal",
            "(i2-carrier-shape? carrier)",
            "(eq? (vector-ref carrier 3)",
            "(i2-native-carrier-live",
            "(g3r-c4-native-owner-binding-status",
            "(vector-ref carrier 2)",
        ):
            self.assertIn(required, WRAPPER)

    def test_native_scalar_contract(self):
        start = OWNER.index("int64_t et_g3c4_private_sealed_owner_binding_status_v1(")
        body = OWNER[start: OWNER.index("int64_t et_g3c4_private_model_owner_abort_v1", start)]
        for required in (
            "et_g3c4_admit_owner(exact_owner, 0)",
            "owner->state != ET_G3C4_OWNER_SEALED",
            "owner->active != NULL",
            "owner->handles[index] != exact_handle",
            "owner->parameters[index] != exact_parameter",
            "et_f32_parameter_validate_identity_v1",
            "ET_G3C4_CODE_STALE_BINDING",
        ):
            self.assertIn(required, body)
        self.assertNotIn("return owner->", body)

    def test_negative_fixture_scope(self):
        for label in ("exact live owner", "cross-owner", "forged root",
                      "swapped handles", "eval mode", "train mode restored",
                      "C2 position [2,4]"):
            self.assertIn(label, FIXTURE)

    def test_changed_live_carrier_restores_and_preserves_bits(self):
        self.assertIn("(define before-registry (vector-ref g3c4-model-registry 0))",
                      FIXTURE)
        self.assertIn("(define before-a-bits (parameter-bytes handles-a))", FIXTURE)
        self.assertIn("(define before-b-bits (parameter-bytes handles-b))", FIXTURE)
        self.assertIn("(vector-set! carrier-a0 2 foreign-parameter)", FIXTURE)
        self.assertEqual(FIXTURE.count("(vector-set! carrier-a0 2 original-parameter)"), 2)
        self.assertIn("(guard (caught", FIXTURE)
        self.assertIn("(status \"changed live P1 carrier rejected\" changed-live-status 4)",
                      FIXTURE)
        self.assertIn("(eq? (vector-ref g3c4-model-registry 0) before-registry)",
                      FIXTURE)
        self.assertIn("(same-parameter-bits? before-a-bits (parameter-bytes handles-a))",
                      FIXTURE)
        self.assertIn("(same-parameter-bits? before-b-bits (parameter-bytes handles-b))",
                      FIXTURE)
        self.assertIn("(status \"association valid after restored carrier\"",
                      FIXTURE)


if __name__ == "__main__":
    unittest.main()
