"""Source-only near misses for C4 capture shape, ownership and runner gates."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "c4_native_capture_check", ROOT / "tests/g3c4/check_c4_native_capture.py")
assert spec and spec.loader
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class NativeCaptureSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.observer = checker.OBSERVER.read_text()
        cls.parser = checker.PARSER.read_text()
        cls.runner = checker.RUNNER.read_text()
        cls.profile = checker.PROFILE.read_text()
        cls.base = checker.BASE.read_text().splitlines()
        cls.closure = checker.CLOSURE.read_text().splitlines()

    def check(self, observer: str | None = None, parser: str | None = None,
              runner: str | None = None, closure: list[str] | None = None,
              profile: str | None = None) -> None:
        checker.check(observer if observer is not None else self.observer,
                      parser if parser is not None else self.parser,
                      runner if runner is not None else self.runner,
                      self.base, closure if closure is not None else self.closure,
                      profile if profile is not None else self.profile)

    def test_current_exact_closure(self) -> None:
        checker.main()

    def test_active_prefix_crop_and_inactive_keep_controls(self) -> None:
        for original, wrong in (
            ("mask[position] == (uint8_t)(position < (size_t)prefix)", "mask[position] != 0"),
            ("(head * 4u + position) * 2u + dimension", "(head * (size_t)prefix + position) * 2u + dimension"),
            ("(head * (size_t)prefix + position) * 2u + dimension", "(head * 4u + position) * 2u + dimension"),
            ("REQUIRE(*(const int64_t *)lengths->data == prefix)", "REQUIRE(prefix <= 4)"),
        ):
            with self.subTest(original=original):
                self.assertIn(original, self.observer)
                with self.assertRaises(AssertionError):
                    self.check(observer=self.observer.replace(original, wrong, 1))

    def test_wrong_t4_abort_identity_rejected(self) -> None:
        marker = "REQUIRE(target_abort_count == 1u && target_abort_result == 0)"
        with self.assertRaises(AssertionError):
            self.check(observer=self.observer.replace(marker,
                "REQUIRE(target_abort_count >= 0u)", 1))

    def test_inherited_closure_and_near_miss_member_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            self.check(closure=self.closure[:-1])
        reordered = self.closure.copy()
        reordered[-1], reordered[-2] = reordered[-2], reordered[-1]
        with self.assertRaises(AssertionError):
            self.check(closure=reordered)
        with self.assertRaises(AssertionError):
            self.check(closure=self.closure + ["tests/g3c4/unrelated.py"])

    def test_q0_provenance_and_repeat_controls(self) -> None:
        for phrase in ("q0_identity", "3.14.6", "2.13.0+cpu",
                       "normal/repeat.stdout", "source.post.sha256",
                       "timeout --signal=TERM --kill-after=5s 600s",
                       "timeout --signal=TERM --kill-after=5s 120s"):
            with self.subTest(phrase=phrase):
                with self.assertRaises(AssertionError):
                    self.check(runner=self.runner.replace(phrase, "removed"))

    def test_status_receipt_does_not_collide_with_canonical_observation(self) -> None:
        self.assertIn('.status.json', self.parser)
        with self.assertRaises(AssertionError):
            self.check(parser=self.parser.replace('.status.json', '.json', 1))

    def test_owner_image_and_rng_checked_before_and_after_each_case(self) -> None:
        for original, wrong in (
            ("require_model_unchanged(owner, seed, image, state);",
             "/* owner comparison omitted */"),
            ("memcmp(current_image, image, 1192u * sizeof(uint32_t)) == 0",
             "memcmp(current_image, image, sizeof(uint32_t)) == 0"),
            ("memcmp(current_state, state, 4u * sizeof(int64_t)) == 0",
             "current_state[0] == state[0]"),
        ):
            with self.subTest(original=original):
                self.assertIn(original, self.observer)
                with self.assertRaises(AssertionError):
                    self.check(observer=self.observer.replace(original, wrong, 1))

    def test_bounded_owned_capture_controls(self) -> None:
        for original in ("MAX_STDERR_BYTES", "selector.select(0.05)",
                         "start_new_session=True", "os.WNOWAIT", "os.wait4",
                         "_signal_session(leader, uid, signal.SIGKILL)"):
            with self.subTest(original=original):
                self.assertIn(original, self.parser)
                with self.assertRaises(AssertionError):
                    self.check(parser=self.parser.replace(original, "removed", 1))
        with self.assertRaises(AssertionError):
            self.check(parser=self.parser + "\nchild.poll()\n")

    def test_external_lock_route_is_explicit_and_narrow(self) -> None:
        for original in ('LOCK_FILE="$candidate_lock"',
                         '[[ "${1:-}" == --candidate-lock-test ]]',
                         'candidate-profile.post.json',
                         'export GIT_OPTIONAL_LOCKS=0',
                         'tests.g3c4.test_c4_toolchain_profile'):
            with self.subTest(original=original):
                self.assertIn(original, self.runner)
                with self.assertRaises(AssertionError):
                    self.check(runner=self.runner.replace(original, "removed", 1))
        with self.assertRaises(AssertionError):
            self.check(runner=self.runner + '\nLOCK_FILE="/unreviewed"\n')
        late = self.runner.replace(
            '  LOCK_FILE="$candidate_lock"\n  export GIT_OPTIONAL_LOCKS=0\nfi\nverify_toolchain\n',
            '  export GIT_OPTIONAL_LOCKS=0\nfi\nverify_toolchain\nLOCK_FILE="$candidate_lock"\n', 1)
        self.assertNotEqual(late, self.runner)
        with self.assertRaises(AssertionError):
            self.check(runner=late)
        for original in ("os.O_NOFOLLOW", "st_nlink == 1",
                         "candidate lock must be outside the tracked C4 checkout",
                         "unsupported-host bypass forbidden", "COMPILER_TREE"):
            with self.subTest(original=original):
                self.assertIn(original, self.profile)
                with self.assertRaises(AssertionError):
                    self.check(profile=self.profile.replace(original, "removed", 1))


if __name__ == "__main__":
    unittest.main()
