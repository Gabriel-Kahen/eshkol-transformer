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

    def test_package_has_no_python_or_torch_runtime_dependency(self) -> None:
        package = (ROOT / "eshkol.toml").read_text(encoding="utf-8").lower()
        self.assertNotIn("python", package)
        self.assertNotIn("torch", package)


if __name__ == "__main__":
    unittest.main()
