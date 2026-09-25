from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q0 = ROOT / "tests" / "q0"
G3C4_DEVELOPMENT_SCRIPTS = frozenset({
    ROOT / "scripts" / name
    for name in (
        "check-g3c4-active-call.py",
        "check-g3c4-call-entry.py",
        "check-g3c4-context-cache.py",
        "check-g3c4-generator.py",
        "check-g3c4-full-prefix-forward.py",
        "check-g3c4-i2-prepared-route.py",
        "check-g3c4-last-logit-frame.py",
        "check-g3c4-model-authority.py",
        "check-g3c4-output-reservation.py",
        "check-g3c4-output-prepare.py",
        "check-g3c4-output-decode-ids.py",
        "check-g3c4-output-text.py",
        "check-g3c4-output-envelope.py",
        "check-g3c4-logits-reservation.py",
        "check-g3c4-manual-frame-begin.py",
        "check-g3c4-manual-role0.py",
        "check-g3c4-manual-pre-a2.py",
        "check-g3c4-manual-a2.py",
        "check-g3c4-manual-at.py",
        "check-g3c4-manual-ao.py",
        "check-g3c4-manual-r.py",
        "check-g3c4-manual-tail.py",
        "check-g3c4-manual-frame-commit.py",
        "check-g3c4-manual-role-step.py",
        "check-g3c4-manual-transport.py",
        "check-g3c4-provider-routes.py",
        "check-g3c4-prefill1.py",
        "check-g3c4-prefill2.py",
        "check-g3c4-prefill3.py",
        "check-g3c4-prompt-prefill.py",
        "check-g3c4-prompt-t1-borrow.py",
        "check-g3c4-sampler-transport.py",
        "check-g3c4-t1-eval-admission.py",
        "check-g3c4-token-frame.py",
        "check-g3c4-token-forward.py",
        "check-p1-prepared-split.py",
    )
})
DEVELOPMENT_SCRIPTS = frozenset({
    ROOT / "scripts" / "generate-e3-d2-source.py",
    ROOT / "scripts" / "check-e3-p1-contract.py",
    ROOT / "scripts" / "check-tr3-p1-fixed.py",
    ROOT / "scripts" / "e3-atomic-publish.py",
    ROOT / "scripts" / "check-g3t-model-admission.py",
    ROOT / "scripts" / "check-g3t-native-context.py",
    ROOT / "scripts" / "check-g3t-generator-constructor.py",
    ROOT / "scripts" / "check-g3t-prefill-sample.py",
    ROOT / "scripts" / "check-g3t-output-text.py",
    ROOT / "scripts" / "check-g3t-final-publication.py",
    ROOT / "scripts" / "check-g3t-zero-budget.py",
    ROOT / "scripts" / "check-g3t-p2-zero-budget.py",
    ROOT / "scripts" / "check-g3t-t1-input.py",
    ROOT / "scripts" / "check-g3t-rng-owner.py",
    ROOT / "scripts" / "check-g3t-generator-rng-input.py",
    ROOT / "scripts" / "check-g3t-owned-token-input.py",
}) | G3C4_DEVELOPMENT_SCRIPTS


def development_python(path: Path) -> bool:
    # Exact development-only build/check helpers, never training dependencies.
    return path.is_relative_to(ROOT / "tests") or path in DEVELOPMENT_SCRIPTS


class PythonIsolationTests(unittest.TestCase):
    def test_python_and_torch_are_test_only(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        python_files = [
            ROOT / item.decode("utf-8")
            for item in tracked.split(b"\0")
            if item
        ]
        for path in python_files:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(
                    development_python(path),
                    f"Python entered an unapproved path {path.relative_to(ROOT)}",
                )
        manifest = (ROOT / "eshkol.toml").read_text(encoding="utf-8")
        self.assertNotIn(".py", manifest)
        self.assertNotIn("torch", manifest.lower())
        self.assertTrue((Q0 / "generate_scalar_add.py").is_file())
        self.assertTrue((Q0 / "requirements-oracle.lock").is_file())

    def test_only_exact_development_scripts_are_admitted_outside_tests(self) -> None:
        for path in DEVELOPMENT_SCRIPTS:
            with self.subTest(path=path):
                self.assertTrue(development_python(path))
        for relative in (
            "scripts/generate-other-source.py", "scripts/runtime.py",
            "scripts/generate-e3-d2-source.py.extra.py",
            "scripts/check-e3-p1-contract.py.extra.py",
            "scripts/e3-atomic-publish.py.extra.py",
            "scripts/nested/generate-e3-d2-source.py",
            "src/generate-e3-d2-source.py", "lib/runtime.py",
            "internal/e3/runtime.py", "native/runtime.py",
        ):
            with self.subTest(path=relative):
                self.assertFalse(development_python(ROOT / relative))

    def test_g3c4_checker_near_misses_are_rejected(self) -> None:
        for admitted in G3C4_DEVELOPMENT_SCRIPTS:
            for path in (
                admitted.with_name(admitted.name + ".extra.py"),
                ROOT / "scripts" / "nested" / admitted.name,
                ROOT / "src" / admitted.name,
                ROOT / "native" / admitted.name,
            ):
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertFalse(development_python(path))

    def test_fixture_reader_has_no_executable_deserialization(self) -> None:
        source = (Q0 / "oracle_format.py").read_text(encoding="utf-8").lower()
        for forbidden in ("pickle", "torch.load", "eval(", "exec(", "importlib"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
