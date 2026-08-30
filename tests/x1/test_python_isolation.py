from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ProductionIsolationTests(unittest.TestCase):
    def test_config_runtime_is_eshkol_only_and_non_executable(self) -> None:
        source = (ROOT / "lib" / "transformer" / "config.esk").read_text(encoding="utf-8")
        forbidden = (
            "(eval ",
            "(load ",
            "get-environment-variable",
            "getenv",
            "process-spawn",
            "subprocess",
            "python",
            "pytorch",
            "torch",
            "include-file",
        )
        lowered = source.lower()
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, lowered)

    def test_public_config_closure_has_only_the_safe_e1b_facade(self) -> None:
        source = (ROOT / "lib" / "transformer" / "config.esk").read_text(encoding="utf-8")
        self.assertIn("(require transformer.error_consumer)", source)
        forbidden = (
            "(require transformer.error_internal)",
            "(require transformer.error_core)",
            "transformer-error-make",
            "transformer-error-raise",
            "transformer-error-wrap-foreign",
            "e1-internal-dispatch",
            "et-e1b-private-raise",
            "x1-make-resolved",
            "x1-resolved-values",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_package_has_no_python_or_torch_runtime_dependency(self) -> None:
        package = (ROOT / "eshkol.toml").read_text(encoding="utf-8").lower()
        self.assertNotIn("python", package)
        self.assertNotIn("torch", package)


if __name__ == "__main__":
    unittest.main()
