import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "native" / "tr3_c_d2_restore_extension.esk"


class Tr3CD2RestoreScopeTests(unittest.TestCase):
    def test_source_closure_is_exact_private_extension(self) -> None:
        base = (ROOT / "native" / "d2_wave2_source_closure.txt").read_text()
        actual = (
            ROOT / "native" / "tr3_c_d2_restore_source_closure.txt"
        ).read_text()
        self.assertEqual(actual, base + "native/tr3_c_d2_restore_extension.esk\n")

    def test_no_public_d2_api_or_export_change(self) -> None:
        public_surfaces = [
            ROOT / "native" / "d2_native.h",
            ROOT / "native" / "d2_wave2_root.esk",
            ROOT / "native" / "d2_wave2_public_exports.txt",
            ROOT / "native" / "d2_wave2_public_strings.txt",
            ROOT / "native" / "d2_wave2_package_bridge.c",
            ROOT / "lib" / "transformer" / "data.esk",
            ROOT / "lib" / "transformer" / "api_contract.esk",
        ]
        for path in public_surfaces:
            with self.subTest(path=path):
                self.assertNotIn("tr3", path.read_text().lower())
        self.assertNotIn(
            "et_tr3_c_d2_dataset_idle_preflight_v1",
            (ROOT / "native" / "d2_native_defined_symbols.txt").read_text(),
        )
        self.assertEqual(
            (
                ROOT / "native" / "tr3_c_d2_restore_defined_symbols.txt"
            ).read_text(),
            "et_tr3_c_d2_dataset_idle_preflight_v1\n",
        )
        native = (ROOT / "native" / "d2_native.c").read_text()
        private = native.index("et_tr3_c_d2_dataset_idle_preflight_v1")
        self.assertNotEqual(native.rfind("#ifdef ET_TR3_C_D2_RESTORE", 0, private), -1)
        self.assertNotEqual(native.find("#endif", private), -1)

    def test_plan_retains_no_cursor_or_epoch_bytes(self) -> None:
        source = EXTENSION.read_text()
        prepare = source[source.index("(define (tr3-d2-restore-prepare-internal") :]
        retained = prepare[prepare.index("(let* ((generation") : prepare.index(
            "(define (tr3-d2-restore-check-internal"
        )]
        for constructor in ("tr3-d2-restore-shadow-tag", "tr3-d2-restore-plan-tag"):
            with self.subTest(constructor=constructor):
                body = retained[retained.index(f"(vector {constructor}") :]
                body = body[: body.index(")))") + 3]
                self.assertNotIn("cursor", body)
                self.assertNotIn("epoch", body)
                self.assertIn("target", body)
                self.assertIn("total", body)
                self.assertIn("precommit", body)
        self.assertNotIn("tr3-d2-restore-plans", source)
        self.assertNotIn("registry-next", source)

    def test_authority_is_authenticated_before_invocation(self) -> None:
        source = EXTENSION.read_text()
        start = source.index("(define (tr3-d2-plan-shadow")
        end = source.index("(define (tr3-d2-plan-admit", start)
        body = source[start:end]
        self.assertLess(
            body.index("(d2-native-shell-factory-validate"),
            body.index("(authority tr3-d2-restore-query)"),
        )
        self.assertIn("(eq? (vector-ref shadow 1) candidate)", body)
        self.assertIn("(eq? (vector-ref shadow 11) authority)", body)

    def test_errors_expose_only_trainer_load_state(self) -> None:
        source = EXTENSION.read_text()
        self.assertNotRegex(source, r"\(operation 'tr3-d2-")
        self.assertEqual(source.count("(operation 'trainer-load-state!)"), 3)

    def test_check_success_path_has_no_allocator_or_public_seek(self) -> None:
        source = EXTENSION.read_text()
        start = source.index("(define (tr3-d2-restore-check-internal")
        end = source.index(";;; Trusted sealed tail", start)
        body = source[start:end]
        for forbidden in (
            "make-vector",
            "make-bytevector",
            "vector-copy",
            "bytevector-copy",
            "string-copy",
            "string->utf8",
            "cons",
            "token-dataset-seek!",
            "d2-core-cursor-decode",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, body)

    def test_commit_is_fixed_nonraising_tail(self) -> None:
        source = EXTENSION.read_text()
        start = source.index("(define (tr3-d2-restore-commit-internal!")
        end = source.index("(define (tr3-d2-restore-abort-internal!", start)
        body = source[start:end]
        normalized = " ".join(body.split())
        self.assertEqual(
            normalized,
            "(define (tr3-d2-restore-commit-internal! plan) "
            "(vector-set! (vector-ref plan 4) 15 (vector-ref plan 9)) "
            "(vector-set! (vector-ref plan 3) 10 'none) "
            "(vector-set! (vector-ref plan 3) 16 #f))",
        )


if __name__ == "__main__":
    unittest.main()
