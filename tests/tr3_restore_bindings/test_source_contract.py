"""Source and isolation contract for private TR3-C restore bindings."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
C = (ROOT / "native/tr3_c_restore_bindings.c").read_text()
H = (ROOT / "native/tr3_c_restore_bindings.h").read_text()
ESK = (ROOT / "native/tr3_c_restore_bindings_extension.esk").read_text()


class RestoreBindingContract(unittest.TestCase):
    def definition(self, name: str, next_name: str) -> str:
        start = ESK.index(f"(define ({name}")
        end = ESK.index(f"(define ({next_name}", start)
        return ESK[start:end]

    def test_fixed_abi_layout_and_exact_entries(self) -> None:
        self.assertIn("ET_TR3_C_RESTORE_REQUEST_BYTES = 728", H)
        self.assertIn("ET_TR3_C_MODEL_DESTINATIONS", C)
        self.assertIn("ET_TR3_C_O2_FIRST_INDEX", C)
        for name in (
            "et_o2_tr3_c_restore_prepare_v1",
            "et_o2_tr3_c_restore_append_v1",
            "et_o2_tr3_c_restore_check_v1",
            "et_o2_tr3_c_restore_commit_native_v1",
            "et_o2_tr3_c_restore_finalize_v1",
            "et_o2_tr3_c_restore_abort_v1",
            "et_tr3_c_i2_copy_builder_create_restore42_v1",
            "et_tr3_c_i2_copy_builder_append_parameter_v1",
            "et_tr3_c_i2_copy_builder_prepare_restore42_v1",
            "et_tr3_c_i2_copy_builder_commit_restore42_checked_v1",
            "et_tr3_c_i2_copy_builder_abort_restore42_checked_v1",
            "et_tr3_c_i2_owned_release_checked_v1",
        ):
            self.assertIn(name, C)

    def test_o2_plan_is_the_only_i2_authority(self) -> None:
        create = self.definition(
            "tr3-c-restore-i2-create-internal",
            "tr3-c-restore-i2-append-parameter-internal!",
        )
        append = self.definition(
            "tr3-c-restore-i2-append-parameter-internal!",
            "tr3-c-restore-o2-append-internal!",
        )
        self.assertIn("(vector-ref (vector-ref canonical 3) 2)", create)
        self.assertIn("(vector-ref o2-ledger 2)", append)
        self.assertNotIn("parent", create + append)

    def test_preallocated_ledger_and_guard_precede_native_ownership(self) -> None:
        for name, next_name, registry, native in (
            ("tr3-c-restore-o2-prepare-internal", "tr3-c-restore-i2-create-internal",
             "o2", "tr3-c-restore-native-o2-prepare"),
            ("tr3-c-restore-i2-create-internal", "tr3-c-restore-i2-append-parameter-internal!",
             "i2", "tr3-c-restore-native-i2-create"),
        ):
            body = self.definition(name, next_name)
            acquire = body.index(f"({native}")
            self.assertLess(body.index(f"(set! tr3-c-restore-{registry}-ledgers next-registry)"), acquire)
            self.assertLess(body.index("(guard (caught"), acquire)
            self.assertLess(body.index(f"(car tr3-c-restore-{registry}-ledgers)"), acquire)
            self.assertNotIn("(tr3-c-restore-find-identity ledger", body)
            after = body[acquire:]
            self.assertNotIn("(cons ", after)
            self.assertNotIn("(vector ", after)
            self.assertNotIn("(make-vector ", after)
            self.assertNotIn("(let transfer", after)

    def test_all_stage_wrappers_transfer_to_existing_output_slot(self) -> None:
        transfer = self.definition("tr3-c-restore-transfer28!", "tr3-c-restore-o2-prepare-internal")
        for index in range(28):
            self.assertIn(f"(vector-set! (vector-ref stage-carriers {index}) 2 '())", transfer)
        self.assertEqual(transfer.count("(vector-set!"), 28)
        self.assertNotIn("(let ", transfer)
        prepare = self.definition("tr3-c-restore-o2-prepare-internal", "tr3-c-restore-i2-create-internal")
        self.assertIn("stages (vector stage-carriers native-request) 'constructing", prepare)
        self.assertEqual(prepare.count("(vector-set! canonical 6 #f)"), 3)
        self.assertEqual(prepare.count("(vector-set! request 6 canonical)"), 2)
        self.assertEqual(prepare.count("(tr3-c-restore-transfer28!"), 4)
        # Both guarded adoption and success publish a prepared exact ledger;
        # the materials root is cleared only after request output publication.
        self.assertEqual(prepare.count("(vector-set! canonical 7 'prepared)"), 2)
        self.assertIn("(eq? (vector-ref request 6) canonical)", prepare)

    def test_actual_region_fixture_covers_canonical_escape_and_drain(self) -> None:
        fixture = (ROOT / "tests/tr3_restore_bindings/source_runtime.esk").read_text()
        self.assertIn("(with-region ('tr3-restore-ledger-rooting 65536)", fixture)
        self.assertIn("regional caller stage wrappers kept raw ownership", fixture)
        self.assertIn("canonical promoted stage wrappers kept raw ownership", fixture)
        self.assertIn("published ledgers did not survive region end", fixture)
        self.assertIn("post-region custody did not drain exactly", fixture)

    def test_native_failure_categories_use_existing_dependency_spelling(self) -> None:
        mapping = self.definition(
            "tr3-c-restore-native-fail", "tr3-c-restore-find-identity"
        )
        for category in (
            "invalid-argument",
            "shape-mismatch",
            "dtype-mismatch",
            "device-mismatch",
            "noncontiguous",
            "unsupported",
            "invalid-state",
            "version-mismatch",
            "corrupt-data",
            "determinism-unavailable",
            "internal",
        ):
            self.assertIn(f"'{category}", mapping)

    def test_irreversible_source_tail_has_no_recoverable_native_mapping(self) -> None:
        i2 = self.definition(
            "tr3-c-restore-i2-commit-tail-internal!",
            "tr3-c-restore-o2-commit-tail-internal!",
        )
        o2 = self.definition(
            "tr3-c-restore-o2-commit-tail-internal!",
            "tr3-c-restore-o2-finalize-tail-internal!",
        )
        finalize = self.definition(
            "tr3-c-restore-o2-finalize-tail-internal!",
            "tr3-c-restore-o2-abort-internal!",
        )
        release = ESK[ESK.index("(define (tr3-c-restore-owned-release-tail-internal!") :]
        self.assertIn("tr3-lease-fail-stop", i2)
        for body in (i2, o2, finalize, release):
            for forbidden in (
                "tr3-c-restore-native-fail",
                "o2-fail",
                "tr3-c-restore-require-i2",
                "tr3-c-restore-require-o2",
                "tr3-c-restore-find-identity",
                "i2-carrier-preflight",
                "tr3-c-restore-owned-carrier",
                "tr3-c-restore-owned-stage",
            ):
                self.assertNotIn(forbidden, body)
        self.assertEqual(i2.count("(if "), 1)
        self.assertNotIn("(if ", o2 + finalize + release)

    def test_no_public_or_generic_handle_surface(self) -> None:
        symbols = set(re.findall(r"\bet_tr3_c_private_[a-z0-9_]+_v1\b", H))
        self.assertEqual(len(symbols), 15)
        for directory in (ROOT / "include", ROOT / "lib", ROOT / "src"):
            for path in directory.rglob("*"):
                if path.is_file():
                    try:
                        text = path.read_text()
                    except UnicodeDecodeError:
                        continue
                    for symbol in symbols:
                        self.assertNotIn(symbol, text, str(path))
        self.assertNotRegex(H, r"get_(?:handle|plan)|export")
        self.assertNotIn("(define (tr3-c-public", ESK)

    def test_feature_gate_requires_the_accepted_restore_union(self) -> None:
        for gate in (
            "ET_TR3_C_RESTORE_BINDINGS",
            "ET_I2_PRIVATE_OWNED_CLONE_MATCH",
            "ET_TR3_C_I2_RESTORE_PRIVATE",
            "ET_TR3_C_O2_RESTORE_NATIVE",
        ):
            self.assertIn(gate, C + H)


if __name__ == "__main__":
    unittest.main()
